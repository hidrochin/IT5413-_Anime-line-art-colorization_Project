"""Optional rank-8 LoRA fine-tune of Model C's U-Net on the training pairs (RQ3 / Experiment 4).

    python -m scripts.train_lora --config configs/controlnet.yaml
    # new Kaggle session, previous session's output attached as input:
    python -m scripts.train_lora --config configs/controlnet.yaml --init-from /kaggle/input/<prev>/controlnet_lora

ControlNet, IP-Adapter, VAE and text encoder stay frozen *in the loop*, so the LoRA learns to color
given exactly the conditioning used at inference. Epsilon-prediction loss at 512 px (targets go
through the same 256->512 upsample as the sketch).

Resumable: <output_dir>/state.pt (LoRA weights, optimizer, scaler, step, data position, RNG states)
is written every `ckpt_every_min` minutes and on exit; the diffusers-format LoRA used by
ControlNetColorizer (`model.lora_path`) is exported to <output_dir>/lora.
"""

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler

from src.models.controlnet.colorizer import ControlNetConfig, build_pipeline, encode_reference, prepare_control, upsample
from src.utils.registry import DEFAULT_DATASET, load_object
from src.utils.seeding import seed_everything


class EpochSampler(Sampler):
    """Seeded permutation per epoch that can start mid-epoch, so a resumed run continues the same order."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_position(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist()[self.start :])

    def __len__(self) -> int:
        return self.n - self.start


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_state(out_dir: Path, unet, optimizer, scaler, progress: dict) -> None:
    from diffusers import StableDiffusionControlNetPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    lora = get_peft_model_state_dict(unet)
    state = {"lora": lora, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "progress": progress, "rng": rng_state()}
    tmp = out_dir / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(out_dir / "state.pt")  # atomic: a session killed mid-save keeps the previous state
    StableDiffusionControlNetPipeline.save_lora_weights(
        out_dir / "lora", unet_lora_layers=convert_state_dict_to_diffusers(lora), safe_serialization=True
    )
    print(f"saved state at step {progress['step']} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/controlnet.yaml"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--init-from", type=Path, default=None, help="folder with a state.pt from an earlier session")
    parser.add_argument("--max-minutes", type=float, default=None, help="save and stop after this long (Kaggle limit)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    lcfg = cfg["lora"]
    mcfg = ControlNetConfig.from_dict({**cfg["model"], "lora_path": None})
    out_dir = Path(lcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "state.pt").exists() and args.init_from is not None:
        shutil.copy(args.init_from / "state.pt", out_dir / "state.pt")
    seed_everything(lcfg["seed"])
    device = torch.device(mcfg.device)

    from diffusers import DDPMScheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    pipe = build_pipeline(mcfg)
    unet, vae, controlnet = pipe.unet, pipe.vae, pipe.controlnet
    for module in (unet, vae, controlnet, pipe.text_encoder, pipe.image_encoder):
        module.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=lcfg["rank"], lora_alpha=lcfg["alpha"], init_lora_weights="gaussian",
                                target_modules=lcfg["target_modules"]))
    cast_training_params(unet, dtype=torch.float32)  # LoRA weights in fp32, frozen base stays fp16
    if lcfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(params, lr=lcfg["learning_rate"], weight_decay=lcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    noise_scheduler = DDPMScheduler.from_pretrained(mcfg.base_model, subfolder="scheduler")
    with torch.no_grad():
        text, _ = pipe.encode_prompt(mcfg.prompt, device, 1, False, clip_skip=mcfg.clip_skip)

    progress = {"step": 0, "epoch": 0, "batch_in_epoch": 0}
    if (out_dir / "state.pt").exists():
        state = torch.load(out_dir / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(unet, state["lora"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        progress = state["progress"]
        print(f"resumed from step {progress['step']} (epoch {progress['epoch']}, batch {progress['batch_in_epoch']})")

    dataset = load_object(args.dataset)("train", "train")
    sampler = EpochSampler(len(dataset), lcfg["seed"])
    bs, accum = lcfg["batch_size"], lcfg["grad_accum"]
    log_path = out_dir / "log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "loss", "elapsed_min"])

    start = last_ckpt = time.time()
    out_of_time = False
    running, micro = [], 0
    while progress["step"] < lcfg["max_steps"] and not out_of_time:
        sampler.set_position(progress["epoch"], progress["batch_in_epoch"] * bs)
        loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=lcfg["num_workers"],
                            drop_last=True, pin_memory=True,
                            generator=torch.Generator().manual_seed(lcfg["seed"] + progress["epoch"]))
        for batch in loader:
            sketch, reference, target = batch["sketch"], batch["reference"], batch["target"]
            with torch.no_grad():
                pixels = upsample(target.to(device).float()).clamp(-1, 1).to(vae.dtype)
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                control = prepare_control(sketch, mcfg.invert_sketch).to(device, controlnet.dtype)
                pos, neg = encode_reference(pipe, reference)
                drop = torch.rand(pos.shape[0], device=pos.device) < lcfg["ref_drop_prob"]
                image_embeds = torch.where(drop.view(-1, *[1] * (pos.ndim - 1)), neg, pos)
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(latents, noise, t)
            text_b = text.expand(latents.shape[0], -1, -1)

            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    down, mid = controlnet(noisy, t, encoder_hidden_states=text_b, controlnet_cond=control,
                                           conditioning_scale=mcfg.controlnet_scale, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=text_b, down_block_additional_residuals=down,
                            mid_block_additional_residual=mid, added_cond_kwargs={"image_embeds": [image_embeds]}).sample
            loss = F.mse_loss(pred.float(), noise.float()) / accum
            scaler.scale(loss).backward()
            running.append(loss.item() * accum)
            micro += 1
            progress["batch_in_epoch"] += 1
            if micro % accum:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, lcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1

            elapsed = (time.time() - start) / 60
            if progress["step"] % lcfg["log_every"] == 0:
                log.writerow([progress["step"], float(np.mean(running)), round(elapsed, 2)])
                log_file.flush()
                print(f"step {progress['step']}: loss {np.mean(running):.4f} ({elapsed:.1f} min)")
                running = []
            out_of_time = args.max_minutes is not None and elapsed >= args.max_minutes
            if out_of_time or progress["step"] >= lcfg["max_steps"]:
                break
            if (time.time() - last_ckpt) / 60 >= lcfg["ckpt_every_min"]:
                save_state(out_dir, unet, optimizer, scaler, progress)
                last_ckpt = time.time()
        else:
            progress["epoch"] += 1
            progress["batch_in_epoch"] = 0

    save_state(out_dir, unet, optimizer, scaler, progress)
    log_file.close()


if __name__ == "__main__":
    main()
