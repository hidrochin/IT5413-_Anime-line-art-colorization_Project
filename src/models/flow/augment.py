"""Batched GPU version of the shared reference augmentation (src/data/warp.py: TPS warp + flip + crop).

On Kaggle's 4 vCPUs the per-image numpy TPS (~0.1-0.2 s each) capped Model B training at ~0.26 it/s with
the GPU mostly idle. Here the loader only decodes the target and the warp runs on the GPU for the whole
batch. Same distribution as `augment_reference` (4x4 lattice, jitter N(0, 0.06 * size), fixed corners,
flip p = 0.5, square crop of 0.8-1 of the side resized back), with bilinear instead of bicubic resampling.
Evaluation references (self_ref) still come from the shared CPU code, so all models see identical inputs.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _tps_kernel(r2: Tensor) -> Tensor:
    return torch.where(r2 > 0, r2 * torch.log(r2.clamp_min(1e-12)), torch.zeros_like(r2))


def tps_solve(src: Tensor, dst: Tensor) -> Tensor:
    """Coefficients (B, n+3, 2) of the TPS mapping dst[i] -> src[i]; src (B,n,2), dst (n,2)."""
    n = dst.shape[0]
    k = _tps_kernel(((dst[:, None] - dst[None]) ** 2).sum(-1))
    p = torch.cat([torch.ones(n, 1, dtype=dst.dtype, device=dst.device), dst], 1)
    a = torch.zeros(n + 3, n + 3, dtype=dst.dtype, device=dst.device)
    a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
    a = a + 1e-8 * torch.eye(n + 3, dtype=dst.dtype, device=dst.device)
    b = torch.zeros(src.shape[0], n + 3, 2, dtype=src.dtype, device=src.device)
    b[:, :n] = src
    return torch.linalg.solve(a.expand(src.shape[0], -1, -1), b)


def tps_apply(coef: Tensor, dst: Tensor, pts: Tensor) -> Tensor:
    """Map points (B, M, 2) through the TPS with control points dst (n,2) -> (B, M, 2)."""
    n = dst.shape[0]
    u = _tps_kernel(((pts[:, :, None] - dst[None, None]) ** 2).sum(-1))  # (B, M, n)
    affine = torch.cat([torch.ones_like(pts[..., :1]), pts], -1)          # (B, M, 3)
    return u @ coef[:, :n] + affine @ coef[:, n:]


def lattice(h: int, w: int, grid: int, device, dtype=torch.float32) -> Tensor:
    ys, xs = torch.meshgrid(torch.linspace(0, h - 1, grid, device=device, dtype=dtype),
                            torch.linspace(0, w - 1, grid, device=device, dtype=dtype), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)  # (grid*grid, 2) as (x, y)


@torch.no_grad()
def augment_reference_gpu(img: Tensor, grid: int = 4, tps_strength: float = 0.06, flip_prob: float = 0.5,
                          min_crop: float = 0.8, generator: torch.Generator | None = None) -> Tensor:
    """(B,3,H,W) ground truth in [-1,1] -> warped, maybe flipped, cropped reference of the same size.

    Built backwards from each output pixel: crop -> flip -> TPS gives the input location to sample."""
    b, _, h, w = img.shape
    dev = img.device
    rand = lambda *shape: torch.rand(*shape, generator=generator, device=dev)  # noqa: E731

    # Crop: an integer square side in [min_crop, 1] of the image, at an integer offset (as in warp.py).
    side = torch.round(min(h, w) * (min_crop + (1 - min_crop) * rand(b)))
    y0 = torch.floor(rand(b) * (h - side + 1))
    x0 = torch.floor(rand(b) * (w - side + 1))
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    scale = (side / w).view(b, 1, 1)
    qx = x0.view(b, 1, 1) + (xs + 0.5) * scale - 0.5  # cv2.resize pixel-centre convention
    qy = y0.view(b, 1, 1) + (ys + 0.5) * (side / h).view(b, 1, 1) - 0.5

    # Flip (applied after the warp in warp.py, so undone first here).
    flip = (rand(b) < flip_prob).view(b, 1, 1)
    qx = torch.where(flip, (w - 1) - qx, qx)

    # TPS: output point dst[i] samples input point src[i]; corners stay fixed.
    dst = lattice(h, w, grid, dev)
    noise = torch.randn(b, grid * grid, 2, generator=generator, device=dev) * (tps_strength * min(h, w))
    corners = torch.tensor([0, grid - 1, grid * (grid - 1), grid * grid - 1], device=dev)
    noise[:, corners] = 0
    coef = tps_solve(dst + noise, dst)
    src = tps_apply(coef, dst, torch.stack([qx, qy], -1).view(b, -1, 2)).view(b, h, w, 2)

    # Sample with border replication (cv2.BORDER_REPLICATE), pixel coords -> [-1, 1].
    norm = torch.stack([src[..., 0] / (w - 1), src[..., 1] / (h - 1)], -1) * 2 - 1
    out = F.grid_sample(img.float(), norm, mode="bilinear", padding_mode="border", align_corners=True)
    return out.clamp(-1, 1).to(img.dtype)
