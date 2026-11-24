"""Model A losses: L_G = LSGAN + lambda_1 * L1 + lambda_p * VGG-19 perceptual (proposal Section 4)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def lsgan_d_loss(d_real: Tensor, d_fake: Tensor) -> Tensor:
    """Least-squares GAN, discriminator side: real -> 1, fake -> 0 (halved, as in pix2pix)."""
    return 0.5 * (F.mse_loss(d_real.float(), torch.ones_like(d_real, dtype=torch.float32))
                  + F.mse_loss(d_fake.float(), torch.zeros_like(d_fake, dtype=torch.float32)))


def lsgan_g_loss(d_fake: Tensor) -> Tensor:
    return F.mse_loss(d_fake.float(), torch.ones_like(d_fake, dtype=torch.float32))


class VGGPerceptualLoss(nn.Module):
    """pix2pixHD's VGG loss: L1 between VGG-19 relu1_1 ... relu5_1 features, weighted 1/32 ... 1.
    Images in [-1,1]; the network is frozen. `features` replaces torchvision's VGG-19 (tests)."""

    LAYERS = (1, 6, 11, 20, 29)   # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 in vgg19().features
    WEIGHTS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)

    def __init__(self, features: nn.Module | None = None):
        super().__init__()
        if features is None:
            from torchvision.models import VGG19_Weights, vgg19

            features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = features[: self.LAYERS[-1] + 1].eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        super().train(mode)
        self.features.eval()  # always frozen
        return self

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = torch.cat([x, y]) * 0.5 + 0.5
        h = (h - self.mean) / self.std
        loss, weights = 0.0, dict(zip(self.LAYERS, self.WEIGHTS))
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in weights:
                fx, fy = h.chunk(2)
                loss = loss + weights[i] * F.l1_loss(fx.float(), fy.detach().float())
        return loss
