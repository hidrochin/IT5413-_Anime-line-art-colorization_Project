"""Distribution metrics: FID and KID on Inception (clean-fid) features, FD on DINOv2 features.

The math matches clean-fid (Fréchet distance; KID = unbiased cubic-kernel MMD over random subsets),
but lives here so it is unit-testable and the KID subsets are seeded.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    sig_a, sig_b = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    diff = mu_a - mu_b
    covmean = linalg.sqrtm(sig_a.dot(sig_b))
    if not np.isfinite(covmean).all():
        offset = np.eye(sig_a.shape[0]) * eps
        covmean = linalg.sqrtm((sig_a + offset).dot(sig_b + offset))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_a) + np.trace(sig_b) - 2 * np.trace(covmean))


def kernel_distance(
    feats_a: np.ndarray, feats_b: np.ndarray, num_subsets: int = 100, max_subset_size: int = 1000, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    n = feats_a.shape[1]
    m = min(feats_a.shape[0], feats_b.shape[0], max_subset_size)
    t = 0.0
    for _ in range(num_subsets):
        x = feats_b[rng.choice(feats_b.shape[0], m, replace=False)]
        y = feats_a[rng.choice(feats_a.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / num_subsets / m)


class InceptionFeatures:
    """clean-fid Inception-V3 pool features (2048-d) with clean resizing."""

    def __init__(self, device: str | torch.device = "cpu", batch_size: int = 64, num_workers: int = 2):
        from cleanfid.fid import build_feature_extractor

        self.device = torch.device(device)
        self.batch_size, self.num_workers = batch_size, num_workers
        self.model = build_feature_extractor("clean", self.device, use_dataparallel=False)

    def __call__(self, files: list[Path]) -> np.ndarray:
        from cleanfid.fid import get_files_features

        return get_files_features(
            [str(f) for f in files], model=self.model, num_workers=self.num_workers,
            batch_size=self.batch_size, device=self.device, mode="clean", verbose=False,
        )


class DinoV2Features:
    """CLS features of DINOv2 (ViT-L/14 by default, as in Stein et al. 2023) at 224 px."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str = "facebook/dinov2-large", device: str | torch.device = "cpu", batch_size: int = 64):
        from transformers import AutoModel

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean, std = torch.tensor(self.MEAN)[:, None, None], torch.tensor(self.STD)[:, None, None]
        return (x - mean) / std

    @torch.no_grad()
    def __call__(self, files: list[Path]) -> np.ndarray:
        feats = []
        for i in range(0, len(files), self.batch_size):
            batch = torch.stack([self._load(f) for f in files[i : i + self.batch_size]]).to(self.device)
            feats.append(self.model(pixel_values=batch).pooler_output.float().cpu().numpy())
        return np.concatenate(feats)
