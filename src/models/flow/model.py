"""Model B network: a latent rectified-flow U-Net conditioned on the sketch latent (channel concat) and
on reference tokens (cross-attention), plus the frozen reference encoder (DINOv2, or CLIP for the ablation).

Convention (design doc, Section 4): t = 0 is noise, t = 1 is data.

    x_t = (1 - t) * eps + t * z1          target velocity  v = z1 - eps
    sampling: Euler on dx/dt = v from t = 0 to 1
"""

from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LATENT_CHANNELS = 4
LATENT_SIZE = 32

# Frozen reference encoders: hub id, token width, and the input size giving the token grid.
REF_ENCODERS = {
    "dinov2": {"id": "facebook/dinov2-small", "dim": 384, "size": 224},     # 16x16 = 256 patch tokens
    "clip": {"id": "openai/clip-vit-base-patch16", "dim": 768, "size": 224},  # 14x14 = 196 patch tokens
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class FlowNetConfig:
    """Architecture of the trainable part. Stored in every checkpoint so sampling rebuilds the same net."""

    ref_encoder: str = "dinov2"
    block_out_channels: list[int] = field(default_factory=lambda: [128, 256, 384, 512])
    layers_per_block: int = 2
    head_dim: int = 64  # channels per attention head (heads per level = channels // head_dim)
    # Cross-attention on the three finest levels (32, 16, 8) and the mid block; plain ResNets at 4x4.
    down_block_types: list[str] = field(default_factory=lambda: [
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"])
    up_block_types: list[str] = field(default_factory=lambda: [
        "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"])

    @classmethod
    def from_dict(cls, d: dict) -> "FlowNetConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown FlowNetConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ref_dim(self) -> int:
        return REF_ENCODERS[self.ref_encoder]["dim"]


class FlowNet(nn.Module):
    """v_theta(x_t, t, z_S, c_R): a diffusers UNet2DConditionModel on [x_t; z_S] (8 channels) predicting a
    4-channel velocity, plus the learned null tokens that replace c_R for classifier-free guidance."""

    def __init__(self, cfg: FlowNetConfig, num_ref_tokens: int = 256):
        super().__init__()
        from diffusers import UNet2DConditionModel

        self.cfg = cfg
        self.unet = UNet2DConditionModel(
            sample_size=LATENT_SIZE,
            in_channels=2 * LATENT_CHANNELS,
            out_channels=LATENT_CHANNELS,
            block_out_channels=tuple(cfg.block_out_channels),
            layers_per_block=cfg.layers_per_block,
            down_block_types=tuple(cfg.down_block_types),
            up_block_types=tuple(cfg.up_block_types),
            cross_attention_dim=cfg.ref_dim,
            # diffusers' legacy naming: `attention_head_dim` is the *number of heads* per level.
            attention_head_dim=tuple(max(1, c // cfg.head_dim) for c in cfg.block_out_channels),
        )
        self.null_tokens = nn.Parameter(torch.randn(1, num_ref_tokens, cfg.ref_dim) * 0.02)

    def null_cond(self, batch: int) -> Tensor:
        return self.null_tokens.expand(batch, -1, -1)

    def forward(self, x_t: Tensor, t: Tensor, z_sketch: Tensor, ref_tokens: Tensor) -> Tensor:
        # diffusers' sinusoidal timestep embedding is tuned for the 0..1000 range.
        return self.unet(torch.cat([x_t, z_sketch], 1), t * 1000, encoder_hidden_states=ref_tokens).sample


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class ReferenceEncoder(nn.Module):
    """Frozen image encoder: (B,3,256,256) in [-1,1] -> patch tokens (B, N, D), CLS token dropped."""

    def __init__(self, kind: str = "dinov2", model=None):
        super().__init__()
        if kind not in REF_ENCODERS:
            raise ValueError(f"unknown ref_encoder {kind!r}; expected one of {sorted(REF_ENCODERS)}")
        self.kind = kind
        spec = REF_ENCODERS[kind]
        self.size = spec["size"]
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if kind == "dinov2" else (CLIP_MEAN, CLIP_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if model is None:
            if kind == "dinov2":
                from transformers import AutoModel

                model = AutoModel.from_pretrained(spec["id"])
            else:
                from transformers import CLIPVisionModel

                model = CLIPVisionModel.from_pretrained(spec["id"])
        self.model = model.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "ReferenceEncoder":
        super().train(mode)
        self.model.eval()  # always frozen
        return self

    @torch.no_grad()
    def forward(self, reference: Tensor) -> Tensor:
        x = (reference.float() + 1) / 2
        x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False, antialias=True)
        x = ((x.clamp(0, 1) - self.mean) / self.std).to(next(self.model.parameters()).dtype)
        hidden = self.model(pixel_values=x).last_hidden_state
        return hidden[:, 1:].float()


def num_ref_tokens(kind: str) -> int:
    patch = 14 if kind == "dinov2" else 16
    return (REF_ENCODERS[kind]["size"] // patch) ** 2


def build_flow_net(cfg: FlowNetConfig) -> FlowNet:
    return FlowNet(cfg, num_ref_tokens(cfg.ref_encoder))


def sample_t(batch: int, device, logit_normal: bool = True, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Training times in (0, 1): logit-normal (SD3) concentrates on the middle of the path."""
    if logit_normal:
        return torch.sigmoid(torch.randn(batch, device=device) * std + mean)
    return torch.rand(batch, device=device)


def flow_matching_loss(net: nn.Module, z1: Tensor, z_sketch: Tensor, ref_tokens: Tensor, t: Tensor,
                       noise: Tensor | None = None) -> Tensor:
    """|| v_theta(x_t, t, z_S, c_R) - (z1 - eps) ||^2, averaged over all elements."""
    eps = torch.randn_like(z1) if noise is None else noise
    tt = t.view(-1, 1, 1, 1)
    x_t = (1 - tt) * eps + tt * z1
    pred = net(x_t, t, z_sketch, ref_tokens)
    return F.mse_loss(pred.float(), (z1 - eps).float())


@torch.no_grad()
def euler_sample(net: FlowNet, noise: Tensor, z_sketch: Tensor, ref_tokens: Tensor, steps: int,
                 guidance: float = 1.0) -> Tensor:
    """Integrate dx/dt = v~ from t = 0 (noise) to t = 1 (data) with `steps` uniform Euler steps.
    v~ = v(null) + w * (v(c_R) - v(null)); w = 1 is the plain conditional model (one pass per step)."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    x = noise
    b = x.shape[0]
    use_cfg = guidance != 1.0
    null = net.null_cond(b).to(ref_tokens.dtype) if use_cfg else None
    ts = torch.linspace(0, 1, steps + 1, device=x.device)
    for i in range(steps):
        t = ts[i].expand(b)
        dt = ts[i + 1] - ts[i]
        if use_cfg:
            v_both = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([z_sketch, z_sketch]),
                         torch.cat([null, ref_tokens]))
            v_null, v_cond = v_both.float().chunk(2)
            v = v_null + guidance * (v_cond - v_null)
        else:
            v = net(x, t, z_sketch, ref_tokens).float()
        x = x + dt * v
    return x
