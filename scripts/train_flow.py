"""Train Model B (latent rectified flow + DINOv2 reference cross-attention) from scratch.

    python -m scripts.cache_latents --splits train val --out data/latents        # once
    python -m scripts.train_flow --config configs/flow.yaml --max-minutes 690
    # next Kaggle session, previous session's output attached as input:
    python -m scripts.train_flow --config configs/flow.yaml --max-minutes 690 \\
        --init-from /kaggle/input/<prev-version>/checkpoints/flow
    # smoke run on 1,000 training images:
    python -m scripts.train_flow --config configs/flow.yaml --limit 1000 --max-steps 3000 --out checkpoints/flow_smoke

Per step: sketch/target latents come from the cache (no VAE in the loop); the loader decodes only the target
image, which is re-warped on the GPU for the whole batch (src/models/flow/augment.py, same distribution as the
shared src/data/warp.py; the CPU warp capped training at ~0.26 it/s on Kaggle) and encoded by the frozen DINOv2. Loss: || v(x_t, t, z_S, c_R) - (z1 - eps) ||^2
with logit-normal t; c_R is replaced by learned null tokens with probability `cond_drop_prob` (for CFG).

Writes to <output_dir>:
    checkpoints/step_*.pt   resumable state (model, EMA, optimizer, scaler, data position, RNG), every
                            `ckpt_every_min` minutes and on exit
    flow_ema.pt             EMA weights + architecture, what configs/flow.yaml `model.checkpoint` points to
    log.csv                 step, loss, val_loss, lr, it_s, elapsed_min
    samples/step_*.png      val grid: sketch | reference | 2 seeds | ground truth
"""

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import ColorizationDataset, find_image, read_image
from src.models.flow.augment import augment_reference_gpu
from src.models.flow.model import (FlowNetConfig, ReferenceEncoder, build_flow_net, euler_sample, flow_matching_loss,
                                   num_params, sample_t)
from src.utils.checkpoint import CheckpointManager, EpochSampler
from src.utils.ema import EMA
from src.utils.image_io import save_png
from src.utils.logging import CSVLogger
from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.seeding import seed_everything
from src.utils.vae import decode, encode, load_vae

DEFAULTS = {
    "output_dir": "checkpoints/flow",
    "latents_dir": "data/latents",
    "batch_size": 64,
    "learning_rate": 1.0e-4,
    "weight_decay": 0.0,
    "warmup_steps": 1000,
    "max_steps": 150000,
    "ema_decay": 0.9999,
    "cond_drop_prob": 0.1,
    "logit_normal": True,
    "max_grad_norm": 1.0,
    "reference_aug": True,   # False = ablation "no warp": the reference is the ground truth itself
    "num_workers": 4,
    "seed": 0,
    "log_every": 100,
    "sample_every": 2000,
    "num_val": 8,
    "val_steps": 16,
    "val_guidance": 2.0,
    "ckpt_every_min": 30,
}


class LatentPairs(Dataset):
    """Training targets + the row of their cached latents. Only the target image is decoded here: the
    sketch comes from the latent cache and the reference warp runs batched on the GPU
    (src/models/flow/augment.py), so the 4 Kaggle CPUs no longer bottleneck training."""

    def __init__(self, base, row_of: dict[str, int]):
        self.base, self.row_of = base, row_of
        inner, self.indices = base, None
        if isinstance(base, Subset):
            inner, self.indices = base.dataset, list(base.indices)
        self.fast = isinstance(inner, ColorizationDataset)
        self.inner = inner

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        if not self.fast:  # e.g. the list-based fake dataset in tests
            x = self.base[idx]
            return {"row": self.row_of[x["id"]], "target": x["target"]}
        image_id = self.inner.ids[self.indices[idx] if self.indices is not None else idx]
        img = read_image(find_image(self.inner.root, "reference", image_id), gray=False)
        return {"row": self.row_of[image_id], "target": torch.from_numpy(img).permute(2, 0, 1).contiguous()}


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """uint8 (0..255) or float [-1,1] images -> float [-1,1]."""
    return x.float() / 127.5 - 1 if x.dtype == torch.uint8 else x.float()


def load_latents(path: Path, device) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m scripts.cache_latents` first")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    row_of = {image_id: i for i, image_id in enumerate(cache["ids"])}
    return row_of, cache["sketch"].to(device), cache["target"].to(device)


def dataset_ids(ds) -> list[str] | None:
    """Ids without reading images: ColorizationDataset.ids, through a Subset if needed."""
    if isinstance(ds, Subset):
        ids = dataset_ids(ds.dataset)
        return None if ids is None else [ids[i] for i in ds.indices]
    ids = getattr(ds, "ids", None)
    return list(ids) if ids is not None else None


def check_ids(base, row_of: dict[str, int]) -> None:
    ids = dataset_ids(base)
    if ids is None:  # e.g. a list-based fake dataset: check by reading the samples
        ids = [base[i]["id"] for i in range(len(base))]
    missing = [i for i in ids if i not in row_of]
    if missing:
        raise KeyError(f"{len(missing)} training ids have no cached latents (e.g. {missing[:3]}); "
                       "re-run scripts.cache_latents with the same splits")


def lr_at(step: int, base_lr: float, warmup: int) -> float:
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


class ValProbe:
    """Fixed val batch: flow loss at fixed (t, eps) for a smooth curve, and a sample grid to eyeball."""

    def __init__(self, dataset, vae, ref_encoder, n: int, device, seed: int = 0):
        items = [dataset[i] for i in range(min(n, len(dataset)))]
        self.sketch = torch.stack([x["sketch"] for x in items])
        self.target = torch.stack([x["target"] for x in items])
        self.reference = torch.stack([x["reference"] for x in items])
        self.vae, self.device = vae, device
        self.z_sketch = encode(vae, self.sketch.to(device))
        self.z1 = encode(vae, self.target.to(device))
        self.tokens = ref_encoder(self.reference.to(device))
        g = torch.Generator().manual_seed(seed)
        self.t = torch.rand(len(items), generator=g).to(device)
        self.noise = [torch.randn(self.z1.shape, generator=g).to(device) for _ in range(2)]

    @torch.no_grad()
    def loss(self, net) -> float:
        net.eval()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            return float(flow_matching_loss(net, self.z1, self.z_sketch, self.tokens, self.t, self.noise[0]))

    @torch.no_grad()
    def grid(self, net, steps: int, guidance: float, path: Path) -> None:
        net.eval()
        outs = []
        for noise in self.noise:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                z = euler_sample(net, noise, self.z_sketch, self.tokens, steps, guidance)
            outs.append(decode(self.vae, z).cpu())
        cols = [self.sketch.repeat(1, 3, 1, 1), self.reference, *outs, self.target]
        rows = [torch.cat([c[i] for c in cols], dim=2) for i in range(len(self.sketch))]
        save_png(torch.cat(rows, dim=1), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/flow.yaml"))
    parser.add_argument("--out", type=Path, default=None, help="overrides train.output_dir")
    parser.add_argument("--latents", type=Path, default=None, help="folder with train.pt (overrides train.latents_dir)")
    parser.add_argument("--init-from", type=Path, default=None, help="checkpoints folder of an earlier session")
    parser.add_argument("--max-minutes", type=float, default=None, help="save and stop after this long (Kaggle limit)")
    parser.add_argument("--max-steps", type=int, default=None, help="overrides train.max_steps")
    parser.add_argument("--limit", type=int, default=None, help="train on the first N training ids only (smoke run)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = {**DEFAULTS, **cfg.get("train", {})}
    unknown = set(tcfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown train keys in {args.config}: {sorted(unknown)}")
    arch = FlowNetConfig.from_dict(cfg.get("arch", {}))
    out_dir = args.out or Path(tcfg["output_dir"])
    max_steps = args.max_steps or tcfg["max_steps"]
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    seed_everything(tcfg["seed"])
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Frozen parts.
    half = torch.float16 if use_amp else torch.float32
    vae = load_vae(cfg.get("model", {}).get("vae", "stabilityai/sd-vae-ft-mse"), str(device), half)
    ref_encoder = ReferenceEncoder(arch.ref_encoder).to(device, half)

    # Trainable part.
    net = build_flow_net(arch).to(device)
    print(f"Model B U-Net: {num_params(net) / 1e6:.1f}M params, reference encoder: {arch.ref_encoder}")
    ema = EMA(net, decay=tcfg["ema_decay"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt = CheckpointManager(out_dir / "checkpoints", every_minutes=tcfg["ckpt_every_min"], init_from=args.init_from)
    state = ckpt.load_latest(model=net, ema=ema, optimizer=optimizer, scaler=scaler)
    step = 0
    build_dataset = load_object(args.dataset)
    base = build_dataset("train", "train")
    if args.limit is not None:
        base = Subset(base, range(min(args.limit, len(base))))
    sampler = EpochSampler(len(base), tcfg["seed"])
    if state is not None:
        if state["arch"] != arch.to_dict():
            raise ValueError(f"checkpoint architecture {state['arch']} != config {arch.to_dict()}")
        step = state["step"]
        sampler.set_position(**state["data"])

    latents_dir = args.latents or Path(tcfg["latents_dir"])
    row_of, z_sketch_all, z_target_all = load_latents(latents_dir / "train.pt", device)
    dataset = LatentPairs(base, row_of)
    check_ids(base, row_of)
    val = ValProbe(build_dataset("val", "self_ref"), vae, ref_encoder, tcfg["num_val"], device)
    log = CSVLogger(out_dir / "log.csv")

    bs = tcfg["batch_size"]
    if len(dataset) < bs:
        raise ValueError(f"only {len(dataset)} training samples for batch size {bs}")
    start = time.time()
    out_of_time = False
    running, t_last, steps_last = [], time.time(), step
    print(f"training from step {step} to {max_steps}, {len(dataset)} samples, batch {bs}")
    while step < max_steps and not out_of_time:
        g = torch.Generator().manual_seed(tcfg["seed"] * 1000003 + sampler.epoch * 1009 + sampler.start)
        loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=tcfg["num_workers"], drop_last=True,
                            pin_memory=use_amp, generator=g, persistent_workers=False)
        for batch in loader:
            net.train()
            rows = batch["row"].to(device)
            z1 = z_target_all[rows].float()
            z_sketch = z_sketch_all[rows].float()
            target = to_unit(batch["target"].to(device, non_blocking=True))
            reference = augment_reference_gpu(target) if tcfg["reference_aug"] else target
            tokens = ref_encoder(reference)
            drop = torch.rand(tokens.shape[0], device=device) < tcfg["cond_drop_prob"]
            tokens = torch.where(drop.view(-1, 1, 1), net.null_cond(tokens.shape[0]).to(tokens.dtype), tokens)
            t = sample_t(z1.shape[0], device, tcfg["logit_normal"])

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, tcfg["learning_rate"], tcfg["warmup_steps"])
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = flow_matching_loss(net, z1, z_sketch, tokens, t)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            ema.update(net)
            step += 1
            sampler.advance(bs)
            running.append(loss.item())

            if step % tcfg["log_every"] == 0 or step == max_steps:
                now = time.time()
                it_s = (step - steps_last) / max(now - t_last, 1e-9)
                val_loss = math.nan
                if step % tcfg["sample_every"] == 0 or step == max_steps:
                    val_loss = val.loss(ema.model)
                    val.grid(ema.model, tcfg["val_steps"], tcfg["val_guidance"], out_dir / "samples" / f"step_{step:08d}.png")
                log.log(step, loss=sum(running) / len(running), val_loss=val_loss,
                        lr=optimizer.param_groups[0]["lr"], it_s=round(it_s, 3))
                running, t_last, steps_last = [], now, step
            out_of_time = args.max_minutes is not None and (time.time() - start) / 60 >= args.max_minutes
            if out_of_time or step >= max_steps:
                break
            if ckpt.due():
                ckpt.save(step, model=net, ema=ema, optimizer=optimizer, scaler=scaler, arch=arch.to_dict(),
                          data=sampler.position())
        else:
            sampler.next_epoch()

    ckpt.save(step, model=net, ema=ema, optimizer=optimizer, scaler=scaler, arch=arch.to_dict(),
              data=sampler.position())
    export = out_dir / "flow_ema.pt"
    tmp = export.with_suffix(".pt.tmp")
    torch.save({"arch": arch.to_dict(), "state_dict": ema.model.state_dict(), "step": step}, tmp)
    tmp.replace(export)
    print(f"stopped at step {step} ({'time limit' if out_of_time else 'max steps'}); EMA weights -> {export}")


if __name__ == "__main__":
    main()
