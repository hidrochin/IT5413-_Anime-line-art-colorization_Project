"""SD-VAE helpers shared by the latent cache, the VAE-ceiling baseline and Model B.

256x256 images <-> 32x32x4 latents, already multiplied by the VAE's scaling factor (0.18215), so
cached latents can be fed to a diffusion/flow model directly.
"""

import torch
from torch import Tensor

VAE_ID = "stabilityai/sd-vae-ft-mse"


def load_vae(model_id: str = VAE_ID, device: str = "cuda", dtype: torch.dtype = torch.float16):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return vae.requires_grad_(False)


@torch.no_grad()
def encode(vae, x: Tensor, sample: bool = False) -> Tensor:
    """(B,C,H,W) in [-1,1], C = 1 (sketch, repeated to RGB) or 3 -> scaled latents (B,4,H/8,W/8), float32.
    `sample=False` uses the posterior mean, which is deterministic and what the cache stores."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    dist = vae.encode(x.to(vae.device, vae.dtype)).latent_dist
    z = dist.sample() if sample else dist.mean
    return z.float() * vae.config.scaling_factor


@torch.no_grad()
def decode(vae, z: Tensor) -> Tensor:
    """Scaled latents -> (B,3,H,W) in [-1,1], float32."""
    x = vae.decode((z / vae.config.scaling_factor).to(vae.device, vae.dtype)).sample
    return x.float().clamp(-1, 1)
