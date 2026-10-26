"""Model C: Stable Diffusion 1.5 + ControlNet (line-art anime) + IP-Adapter, zero-shot or with a LoRA.

SD1.5 is native 512 px, so this is the one documented resolution exception (fairness rule 1):
the 256 sketch is upsampled to 512 for the ControlNet and the 512 output is downsampled back to 256,
with the same two resize ops (`upsample`, `downsample`) for every image.
"""

from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.colorizer import Colorizer

INTERNAL_SIZE = 512
OUTPUT_SIZE = 256


@dataclass
class ControlNetConfig:
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet: str = "lllyasviel/control_v11p_sd15s2_lineart_anime"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"
    ip_adapter_scale: float = 0.6
    controlnet_scale: float = 1.0
    guidance_scale: float = 7.0
    steps: int = 20
    # The line-art-anime ControlNet was trained with CLIP skip 2 (diffusers clip_skip=1).
    clip_skip: int | None = 1
    prompt: str = "anime character face, flat colors, best quality"
    negative_prompt: str = "lowres, worst quality, low quality, blurry, monochrome, grayscale"
    # Our sketches are dark lines on white; the ControlNet expects white lines on black.
    invert_sketch: bool = True
    lora_path: str | None = None
    lora_scale: float = 1.0
    dtype: str = "float16"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "ControlNetConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ControlNetConfig keys: {sorted(unknown)}")
        return cls(**d)


def upsample(x: Tensor) -> Tensor:
    """256 -> 512 for the ControlNet input."""
    return F.interpolate(x, size=(INTERNAL_SIZE, INTERNAL_SIZE), mode="bicubic", align_corners=False)


def downsample(x: Tensor) -> Tensor:
    """512 -> 256 for the output, antialiased."""
    return F.interpolate(x, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bicubic", align_corners=False, antialias=True)


def prepare_control(sketch: Tensor, invert: bool = True) -> Tensor:
    """(B,1,256,256) in [-1,1] -> (B,3,512,512) in [0,1], white lines on black if `invert`."""
    x = (sketch.float() + 1) / 2
    if invert:
        x = 1 - x
    return upsample(x).clamp(0, 1).repeat(1, 3, 1, 1)


def to_pil(images: Tensor) -> list[Image.Image]:
    """(B,3,H,W) in [-1,1] -> list of PIL images."""
    arr = ((images.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(a) for a in arr.permute(0, 2, 3, 1).numpy()]


def build_pipeline(cfg: ControlNetConfig):
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline

    dtype = getattr(torch, cfg.dtype)
    controlnet = ControlNetModel.from_pretrained(cfg.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.base_model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
    pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder=cfg.ip_adapter_subfolder, weight_name=cfg.ip_adapter_weight)
    if cfg.lora_path:
        pipe.load_lora_weights(cfg.lora_path, adapter_name="colorize")
    pipe.to(cfg.device)
    # Don't call enable_attention_slicing(): it swaps the attention processors and silently drops
    # IP-Adapter. PyTorch 2 SDPA is memory-efficient enough for a T4.
    check_ip_adapter(pipe.unet)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_reference(pipe, reference: Tensor) -> tuple[Tensor, Tensor]:
    """(B,3,H,W) in [-1,1] -> (positive, negative) IP-Adapter embeddings, each (B, 1, ...).
    Shared by inference and LoRA training so both see identical reference features."""
    from diffusers.models.embeddings import ImageProjection

    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    hidden = not isinstance(proj, ImageProjection)  # IP-Adapter Plus uses penultimate hidden states
    pos, neg = pipe.encode_image(to_pil(reference), pipe.device, 1, hidden)
    return pos.unsqueeze(1), neg.unsqueeze(1)


def check_ip_adapter(unet) -> None:
    if not any("IPAdapter" in type(p).__name__ for p in unet.attn_processors.values()):
        raise RuntimeError("IP-Adapter attention processors are missing: the reference would be ignored")


class ControlNetColorizer(Colorizer):
    name = "controlnet"

    def __init__(self, cfg: ControlNetConfig | dict | None = None, pipe=None):
        if isinstance(cfg, dict):
            cfg = ControlNetConfig.from_dict(cfg)
        self.cfg = cfg or ControlNetConfig()
        self.pipe = pipe if pipe is not None else build_pipeline(self.cfg)
        self.set_params(ip_adapter_scale=self.cfg.ip_adapter_scale, lora_scale=self.cfg.lora_scale)

    def set_params(self, **kwargs) -> None:
        """Change sampling-time settings without reloading weights (used by tune_controlnet.py)."""
        for key, value in kwargs.items():
            if key in ("base_model", "controlnet", "ip_adapter_repo", "ip_adapter_weight", "lora_path", "dtype"):
                raise ValueError(f"{key} needs a new pipeline; build a new ControlNetColorizer instead")
            setattr(self.cfg, key, value)
        if "ip_adapter_scale" in kwargs:
            self.pipe.set_ip_adapter_scale(self.cfg.ip_adapter_scale)
        if "lora_scale" in kwargs and self.cfg.lora_path:
            self.pipe.set_adapters(["colorize"], [self.cfg.lora_scale])

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    @property
    def do_cfg(self) -> bool:
        return self.cfg.guidance_scale > 1.0

    def ip_adapter_embeds(self, reference: Tensor) -> list[Tensor]:
        """Per-sample embeddings in the pipeline's `ip_adapter_image_embeds` format: a list (one entry
        per IP-Adapter) of (B, 1, ...) tensors, negatives first when CFG is on. Passing images instead
        would make every sample attend to every reference in the batch."""
        pos, neg = encode_reference(self.pipe, reference)
        return [torch.cat([neg, pos]) if self.do_cfg else pos]

    @torch.no_grad()
    def colorize(self, sketch: Tensor, reference: Tensor, seed: int, steps: int | None = None) -> Tensor:
        b = sketch.shape[0]
        # One CPU generator per sample, all with the same seed: a sample's output depends only on
        # (sketch, reference, seed), not on its batch neighbours or the GPU model.
        generators = [torch.Generator("cpu").manual_seed(seed) for _ in range(b)]
        images = self.pipe(
            prompt=[self.cfg.prompt] * b,
            negative_prompt=[self.cfg.negative_prompt] * b,
            image=prepare_control(sketch, self.cfg.invert_sketch),
            ip_adapter_image_embeds=self.ip_adapter_embeds(reference),
            num_inference_steps=steps or self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            controlnet_conditioning_scale=self.cfg.controlnet_scale,
            clip_skip=self.cfg.clip_skip,
            height=INTERNAL_SIZE,
            width=INTERNAL_SIZE,
            generator=generators,
            output_type="pt",
        ).images
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).permute(0, 3, 1, 2)
        out = downsample(images.float().cpu()).clamp(0, 1) * 2 - 1
        return out.to(sketch.device)

    @classmethod
    def from_config(cls, cfg: dict) -> "ControlNetColorizer":
        """Build from the `model:` section of configs/controlnet.yaml."""
        return cls(ControlNetConfig.from_dict(cfg))
