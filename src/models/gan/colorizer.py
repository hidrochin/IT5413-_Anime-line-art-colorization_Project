"""Model A: reference-conditioned pix2pix GAN behind the shared Colorizer interface.

    [sketch; warped reference] --U-Net G (dropout on, seeded per sample)--> colored image   (1 pass)

Weights come from scripts/train_gan.py: the exported `gan_best.pt` (best val LPIPS) / `gan_G.pt` (last step),
or a training checkpoint (`step_*.pt`, or the folder holding them, in which case the newest is used).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import Tensor

from src.colorizer import Colorizer
from src.models.gan.networks import GanNetConfig, UnetGenerator, build_generator, seeded_dropout
from src.utils.checkpoint import list_checkpoints


@dataclass
class GanConfig:
    checkpoint: str | None = None
    test_dropout: bool = True   # False: plain eval mode, every seed gives the same image (ablation)
    dtype: str = "float32"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "GanConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown GanConfig keys: {sorted(unknown)}")
        return cls(**d)


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in ("gan_best.pt", "gan_G.pt"):
            if (path / name).exists():
                return path / name
        found = list_checkpoints(path)
        if not found:
            raise FileNotFoundError(f"no gan_best.pt, gan_G.pt or step_*.pt in {path}")
        return found[-1][1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_gan_weights(path: str | Path) -> tuple[GanNetConfig, dict, int]:
    """(architecture, generator state dict, step) from an exported file or a CheckpointManager file."""
    payload = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    if "objects" in payload:  # CheckpointManager format
        objects = payload["objects"]
        return GanNetConfig.from_dict(objects["arch"]), objects["G"], payload["step"]
    return GanNetConfig.from_dict(payload["arch"]), payload["state_dict"], payload.get("step", -1)


def dropout_generators(batch: int, seed: int) -> list[torch.Generator]:
    """One CPU generator per sample, all seeded with `seed`: a sample's dropout masks depend only on the seed."""
    return [torch.Generator("cpu").manual_seed(seed) for _ in range(batch)]


class GanColorizer(Colorizer):
    name = "gan"

    def __init__(self, cfg: GanConfig | dict | None = None, net: UnetGenerator | None = None):
        if isinstance(cfg, dict):
            cfg = GanConfig.from_dict(cfg)
        self.cfg = cfg or GanConfig()
        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype) if self.device.type == "cuda" else torch.float32
        self.step = -1
        if net is None:
            if not self.cfg.checkpoint:
                raise ValueError("GanConfig.checkpoint is required (set model.checkpoint in configs/gan.yaml)")
            arch, state, self.step = load_gan_weights(self.cfg.checkpoint)
            net = build_generator(arch)
            net.load_state_dict(state)
            print(f"loaded Model A weights from {self.cfg.checkpoint} (step {self.step})")
        # eval(): BatchNorm uses its running statistics, so a sample never depends on its batch neighbours.
        self.net = net.to(self.device, self.dtype).eval().requires_grad_(False)

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    @torch.no_grad()
    def colorize(self, sketch: Tensor, reference: Tensor, seed: int, steps: int | None = None) -> Tensor:
        gens = dropout_generators(sketch.shape[0], seed) if self.cfg.test_dropout else None
        with seeded_dropout(self.net, gens):
            out = self.net(sketch.to(self.device, self.dtype), reference.to(self.device, self.dtype))
        return out.float().clamp(-1, 1).to(sketch.device)

    @classmethod
    def from_config(cls, cfg: dict) -> "GanColorizer":
        """Build from the `model:` section of configs/gan.yaml."""
        return cls(GanConfig.from_dict(cfg))
