"""Train Model A (reference-conditioned pix2pix GAN) from scratch.

    python -m scripts.train_gan --config configs/gan.yaml --max-minutes 690
    # next Kaggle session, previous session's output attached as input:
    python -m scripts.train_gan --config configs/gan.yaml --max-minutes 690 \\
        --init-from /kaggle/input/<prev-version>/gan/checkpoints
    # smoke run on 1,000 training images:
    python -m scripts.train_gan --config configs/gan.yaml --limit 1000 --max-steps 2000 --out checkpoints/gan_smoke

Per step: the loader decodes sketch + target only; the reference is the target re-warped on the GPU for the
whole batch (src/models/flow/augment.py, same distribution as the shared src/data/warp.py, which is too slow
on Kaggle's 4 vCPUs). Then one D step (LSGAN on [S; Y] vs [S; G(S,R)]) and one G step
(LSGAN + lambda_l1 * L1 + lambda_vgg * VGG-19). Learning rate: constant, then linear to 0 from
`lr_decay_start` to `max_steps` (pix2pix schedule).

Checkpoint selection on val (fairness rule 4): every `val_every` steps, LPIPS to GT on the first `num_val`
val images (self_ref references, dropout seed 0, as at test time); the best generator is exported.

Writes to <output_dir>:
    checkpoints/step_*.pt   resumable state (G, D, both optimizers and scalers, data position, best, RNG)
    gan_best.pt             generator with the lowest val LPIPS (what configs/gan.yaml `model.checkpoint` points to)
    gan_G.pt                generator at the last step
    log.csv                 step, loss_d, loss_g_adv, loss_l1, loss_vgg, val_l1, val_lpips, lr, it_s, elapsed_min
    samples/step_*.png      val grid: sketch | reference | dropout seed 0 | seed 1 | ground truth
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import ColorizationDataset, find_image, read_image, to_tensor
from src.data.sketch import normalize_sketch
from src.metrics.perceptual import LPIPS
from src.models.flow.augment import augment_reference_gpu
from src.models.gan.colorizer import dropout_generators
from src.models.gan.losses import VGGPerceptualLoss, lsgan_d_loss, lsgan_g_loss
from src.models.gan.networks import GanNetConfig, build_discriminator, build_generator, num_params, seeded_dropout
from src.utils.checkpoint import CheckpointManager, EpochSampler
from src.utils.image_io import save_png
from src.utils.logging import CSVLogger
from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.seeding import seed_everything

DEFAULTS = {
    "output_dir": "checkpoints/gan",
    "batch_size": 16,
    "learning_rate": 2.0e-4,
    "betas": [0.5, 0.999],
    "max_steps": 60000,
    "lr_decay_start": 30000,
    "lambda_l1": 100.0,
    "lambda_vgg": 10.0,
    "gpu_warp": True,        # False: use the shared CPU warp from the Dataset's train mode (slow on Kaggle)
    "reference_aug": True,   # False = ablation "no warp": the reference is the ground truth itself
    "amp": True,
    "num_workers": 4,
    "seed": 0,
    "log_every": 100,
    "val_every": 2000,
    "num_val": 200,
    "val_batch": 16,
    "ckpt_every_min": 30,
}


class SketchTargetPairs(Dataset):
    """Training sketch + target without building a reference (it is warped on the GPU). Falls back to the
    wrapped dataset's own samples for anything that isn't a ColorizationDataset (e.g. the test fakes)."""

    def __init__(self, base):
        self.base = base
        inner, self.indices = base, None
        if isinstance(base, Subset):
            inner, self.indices = base.dataset, list(base.indices)
        self.fast = isinstance(inner, ColorizationDataset)
        self.inner = inner

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        if not self.fast:
            x = self.base[idx]
            return {"sketch": x["sketch"], "target": x["target"]}
        image_id = self.inner.ids[self.indices[idx] if self.indices is not None else idx]
        sketch = read_image(find_image(self.inner.root, "sketch", image_id), gray=True)
        target = read_image(find_image(self.inner.root, "reference", image_id), gray=False)
        return {"sketch": to_tensor(normalize_sketch(sketch, self.inner.sketch_style)), "target": to_tensor(target)}


def lr_at(step: int, base_lr: float, decay_start: int, max_steps: int) -> float:
    if step < decay_start:
        return base_lr
    return base_lr * max(0.0, (max_steps - step) / max(1, max_steps - decay_start))


class ValProbe:
    """Fixed val images: L1 and LPIPS to GT with dropout seed 0 (the test protocol), and a sample grid."""

    def __init__(self, dataset, n: int, batch: int, device, lpips_fn):
        items = [dataset[i] for i in range(min(n, len(dataset)))]
        self.sketch = torch.stack([x["sketch"] for x in items])
        self.target = torch.stack([x["target"] for x in items])
        self.reference = torch.stack([x["reference"] for x in items])
        self.batch, self.device, self.lpips = batch, device, lpips_fn

    @torch.no_grad()
    def run(self, net, seed: int, n: int | None = None) -> torch.Tensor:
        net.eval()
        outs = []
        for k in range(0, len(self.sketch) if n is None else n, self.batch):
            s, r = self.sketch[k : k + self.batch], self.reference[k : k + self.batch]
            with seeded_dropout(net, dropout_generators(len(s), seed)):
                outs.append(net(s.to(self.device), r.to(self.device)).float().cpu())
        net.train()
        return torch.cat(outs)

    def scores(self, net) -> dict:
        pred = self.run(net, seed=0)
        return {"val_l1": F.l1_loss(pred, self.target).item(),
                "val_lpips": self.lpips((pred + 1) / 2, (self.target + 1) / 2).mean().item()}

    def grid(self, net, path: Path, n: int = 8) -> None:
        n = min(n, len(self.sketch))
        cols = [self.sketch[:n].repeat(1, 3, 1, 1), self.reference[:n], self.run(net, 0, n)[:n], self.run(net, 1, n)[:n],
                self.target[:n]]
        rows = [torch.cat([c[i] for c in cols], dim=2) for i in range(n)]
        save_png(torch.cat(rows, dim=1), path)


def export(net, arch: GanNetConfig, step: int, path: Path, **extra) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"arch": arch.to_dict(), "state_dict": net.state_dict(), "step": step, **extra}, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/gan.yaml"))
    parser.add_argument("--out", type=Path, default=None, help="overrides train.output_dir")
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
    arch = GanNetConfig.from_dict(cfg.get("arch", {}))
    out_dir = args.out or Path(tcfg["output_dir"])
    max_steps = args.max_steps or tcfg["max_steps"]
    decay_start = min(tcfg["lr_decay_start"], max_steps)
    device = torch.device(args.device)
    use_amp = tcfg["amp"] and device.type == "cuda"
    seed_everything(tcfg["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    net_g, net_d = build_generator(arch).to(device), build_discriminator(arch).to(device)
    vgg = VGGPerceptualLoss().to(device)
    print(f"Model A: G {num_params(net_g) / 1e6:.1f}M params, D {num_params(net_d) / 1e6:.1f}M params")
    opt_g = torch.optim.Adam(net_g.parameters(), lr=tcfg["learning_rate"], betas=tuple(tcfg["betas"]))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=tcfg["learning_rate"], betas=tuple(tcfg["betas"]))
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt = CheckpointManager(out_dir / "checkpoints", every_minutes=tcfg["ckpt_every_min"], init_from=args.init_from)
    state = ckpt.load_latest(G=net_g, D=net_d, opt_g=opt_g, opt_d=opt_d, scaler_g=scaler_g, scaler_d=scaler_d)
    build_dataset = load_object(args.dataset)
    base = build_dataset("train", "train")
    if args.limit is not None:
        base = Subset(base, range(min(args.limit, len(base))))
    dataset = SketchTargetPairs(base) if tcfg["gpu_warp"] else base
    sampler = EpochSampler(len(dataset), tcfg["seed"])
    step, best = 0, {"val_lpips": math.inf, "step": -1}
    if state is not None:
        if state["arch"] != arch.to_dict():
            raise ValueError(f"checkpoint architecture {state['arch']} != config {arch.to_dict()}")
        step, best = state["step"], state["best"]
        sampler.set_position(**state["data"])

    val = ValProbe(build_dataset("val", "self_ref"), tcfg["num_val"], tcfg["val_batch"], device,
                   LPIPS(device=device))
    log = CSVLogger(out_dir / "log.csv")

    def save() -> None:
        ckpt.save(step, G=net_g, D=net_d, opt_g=opt_g, opt_d=opt_d, scaler_g=scaler_g, scaler_d=scaler_d,
                  arch=arch.to_dict(), data=sampler.position(), best=best)

    bs = tcfg["batch_size"]
    if len(dataset) < bs:
        raise ValueError(f"only {len(dataset)} training samples for batch size {bs}")
    start = time.time()
    out_of_time = False
    running: dict[str, list[float]] = {}
    t_last, steps_last = time.time(), step
    print(f"training from step {step} to {max_steps}, {len(dataset)} samples, batch {bs}")
    net_g.train()
    net_d.train()
    while step < max_steps and not out_of_time:
        g = torch.Generator().manual_seed(tcfg["seed"] * 1000003 + sampler.epoch * 1009 + sampler.start)
        loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=tcfg["num_workers"], drop_last=True,
                            pin_memory=device.type == "cuda", generator=g, persistent_workers=False)
        for batch in loader:
            sketch = batch["sketch"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if not tcfg["reference_aug"]:
                reference = target
            elif tcfg["gpu_warp"]:
                reference = augment_reference_gpu(target)
            else:
                reference = batch["reference"].to(device, non_blocking=True)
            lr = lr_at(step, tcfg["learning_rate"], decay_start, max_steps)
            for opt in (opt_g, opt_d):
                for group in opt.param_groups:
                    group["lr"] = lr

            # D step: real [S; Y] -> 1, fake [S; G(S,R)] -> 0.
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                fake = net_g(sketch, reference)
                loss_d = lsgan_d_loss(net_d(sketch, target), net_d(sketch, fake.detach()))
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # G step: fool D + stay close to Y in pixels and VGG features.
            net_d.requires_grad_(False)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                adv = lsgan_g_loss(net_d(sketch, fake))
                l1 = F.l1_loss(fake.float(), target.float())
                perc = vgg(fake, target)
                loss_g = adv + tcfg["lambda_l1"] * l1 + tcfg["lambda_vgg"] * perc
            net_d.requires_grad_(True)
            if not (torch.isfinite(loss_g) and torch.isfinite(loss_d)):
                raise FloatingPointError(f"non-finite loss at step {step}: G {loss_g.item()}, D {loss_d.item()}")
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            step += 1
            sampler.advance(bs)
            for k, v in (("loss_d", loss_d), ("loss_g_adv", adv), ("loss_l1", l1), ("loss_vgg", perc)):
                running.setdefault(k, []).append(v.detach().item())

            if step % tcfg["log_every"] == 0 or step == max_steps:
                now = time.time()
                it_s = (step - steps_last) / max(now - t_last, 1e-9)
                scores = {"val_l1": math.nan, "val_lpips": math.nan}
                if step % tcfg["val_every"] == 0 or step == max_steps:
                    scores = val.scores(net_g)
                    val.grid(net_g, out_dir / "samples" / f"step_{step:08d}.png")
                    if scores["val_lpips"] < best["val_lpips"]:
                        best = {"val_lpips": scores["val_lpips"], "step": step}
                        export(net_g, arch, step, out_dir / "gan_best.pt", val_lpips=scores["val_lpips"])
                log.log(step, **{k: sum(v) / len(v) for k, v in running.items()}, **scores, lr=lr,
                        it_s=round(it_s, 3))
                running, t_last, steps_last = {}, now, step
            out_of_time = args.max_minutes is not None and (time.time() - start) / 60 >= args.max_minutes
            if out_of_time or step >= max_steps:
                break
            if ckpt.due():
                save()
        else:
            sampler.next_epoch()

    save()
    export(net_g, arch, step, out_dir / "gan_G.pt")
    print(f"stopped at step {step} ({'time limit' if out_of_time else 'max steps'}); last G -> {out_dir / 'gan_G.pt'}, "
          f"best val LPIPS {best['val_lpips']:.4f} at step {best['step']} -> {out_dir / 'gan_best.pt'}")


if __name__ == "__main__":
    main()
