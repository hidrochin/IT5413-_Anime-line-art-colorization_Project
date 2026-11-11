"""Model B: latent rectified flow with DINOv2 reference cross-attention, behind the shared Colorizer interface.

    sketch --VAE--> z_S ┐
                        ├─ U-Net (v_theta) ◄── cross-attention ── c_R ◄── DINOv2 ── reference
    noise eps ──────────┘   │ Euler, N steps, CFG weight w
                            ▼
                           z1 --VAE decode--> colored image

Weights come from scripts/train_flow.py: either the exported `flow_ema.pt` or a training checkpoint
(`step_*.pt`, or the folder holding them, in which case the newest is used).
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import Tensor

from src.colorizer import Colorizer
from src.models.flow.model import (LATENT_CHANNELS, LATENT_SIZE, FlowNet, FlowNetConfig, ReferenceEncoder,
                                   build_flow_net, euler_sample)
from src.utils.checkpoint import list_checkpoints
from src.utils.vae import VAE_ID


@dataclass
class FlowConfig:
    checkpoint: str | None = None
    use_ema: bool = True
    steps: int = 32
    guidance_scale: float = 2.0
    vae: str = VAE_ID
    dtype: str = "float16"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "FlowConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown FlowConfig keys: {sorted(unknown)}")
        return cls(**d)


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        if (path / "flow_ema.pt").exists():
            return path / "flow_ema.pt"
        found = list_checkpoints(path)
        if not found:
            raise FileNotFoundError(f"no flow_ema.pt or step_*.pt in {path}")
        return found[-1][1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_flow_weights(path: str | Path, use_ema: bool = True) -> tuple[FlowNetConfig, dict, int]:
    """(architecture, U-Net state dict, step) from an exported file or a CheckpointManager file."""
    payload = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    if "objects" in payload:  # CheckpointManager format
        objects = payload["objects"]
        state = objects["ema"]["model"] if use_ema else objects["model"]
        return FlowNetConfig.from_dict(objects["arch"]), state, payload["step"]
    return FlowNetConfig.from_dict(payload["arch"]), payload["state_dict"], payload.get("step", -1)


def initial_noise(batch: int, seed: int) -> Tensor:
    """One CPU generator per sample, all seeded with `seed`: a sample's output depends only on
    (sketch, reference, seed), not on its batch neighbours or the GPU."""
    shape = (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    return torch.stack([torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed)) for _ in range(batch)])


class FlowColorizer(Colorizer):
    name = "flow"

    def __init__(self, cfg: FlowConfig | dict | None = None, net: FlowNet | None = None, vae=None,
                 ref_encoder: ReferenceEncoder | None = None):
        if isinstance(cfg, dict):
            cfg = FlowConfig.from_dict(cfg)
        self.cfg = cfg or FlowConfig()
        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype) if self.device.type == "cuda" else torch.float32
        self.step = -1
        if net is None:
            if not self.cfg.checkpoint:
                raise ValueError("FlowConfig.checkpoint is required (set model.checkpoint in configs/flow.yaml)")
            arch, state, self.step = load_flow_weights(self.cfg.checkpoint, self.cfg.use_ema)
            net = build_flow_net(arch)
            net.load_state_dict(state)
            print(f"loaded Model B weights from {self.cfg.checkpoint} (step {self.step}, ema={self.cfg.use_ema})")
        self.net = net.to(self.device, self.dtype).eval().requires_grad_(False)
        if vae is None:
            from src.utils.vae import load_vae

            vae = load_vae(self.cfg.vae, str(self.device), self.dtype)
        self.vae = vae
        if ref_encoder is None:
            ref_encoder = ReferenceEncoder(self.net.cfg.ref_encoder)
        self.ref_encoder = ref_encoder.to(self.device, self.dtype)

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def nfe(self, steps: int | None = None) -> int:
        """U-Net passes per image: CFG (w != 1) runs the conditional and null branch every step."""
        return (steps or self.cfg.steps) * (2 if self.cfg.guidance_scale != 1.0 else 1)

    @torch.no_grad()
    def colorize(self, sketch: Tensor, reference: Tensor, seed: int, steps: int | None = None) -> Tensor:
        from src.utils.vae import decode, encode

        z_sketch = encode(self.vae, sketch.to(self.device)).to(self.dtype)
        ref_tokens = self.ref_encoder(reference.to(self.device)).to(self.dtype)
        noise = initial_noise(sketch.shape[0], seed).to(self.device, self.dtype)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            z1 = euler_sample(self.net, noise, z_sketch, ref_tokens, steps or self.cfg.steps,
                              self.cfg.guidance_scale)
        return decode(self.vae, z1).to(sketch.device)

    @classmethod
    def from_config(cls, cfg: dict) -> "FlowColorizer":
        """Build from the `model:` section of configs/flow.yaml."""
        return cls(FlowConfig.from_dict(cfg))
