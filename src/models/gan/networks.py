"""Model A networks: the pix2pix U-Net generator and 70x70 PatchGAN discriminator (Isola et al., 2017),
following the official pytorch-CycleGAN-and-pix2pix code (UnetGenerator / NLayerDiscriminator, BatchNorm,
normal(0, 0.02) init).

    G: [S; warped R] (4 ch) -> Y_hat (3 ch, tanh)     8 down/up levels with skips, dropout in 3 inner decoders
    D: [S; Y] (1 + 3 ch) -> 30x30 real/fake map      each output sees a 70x70 patch

Dropout is the only source of randomness, so it stays on at test time. `seeded_dropout` draws every
sample's masks from its own CPU generator: an output then depends only on (input, seed), not on its
batch neighbours or the device (same convention as Models B and C).
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn

SKETCH_CHANNELS = 1
IMAGE_CHANNELS = 3


@dataclass
class GanNetConfig:
    """Architecture, stored in every checkpoint so sampling rebuilds the same network."""

    ngf: int = 64          # generator width at the outermost level
    ndf: int = 64          # discriminator width
    num_downs: int = 8     # 256 -> 1x1 bottleneck
    n_layers_d: int = 3    # 3 -> 70x70 receptive field
    dropout: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "GanNetConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown GanNetConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


class SeededDropout(nn.Dropout):
    """nn.Dropout that, while `generators` is set (one CPU generator per sample), draws each sample's mask from
    its own generator, in train or eval mode. Without generators it is a plain nn.Dropout."""

    generators: list[torch.Generator] | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.generators is None:
            return super().forward(x)
        if len(self.generators) != x.shape[0]:
            raise ValueError(f"{len(self.generators)} generators for a batch of {x.shape[0]}")
        keep = torch.stack([torch.rand(x.shape[1:], generator=g) for g in self.generators]) >= self.p
        return x * keep.to(x.device, x.dtype) / (1 - self.p)


@contextmanager
def seeded_dropout(net: nn.Module, generators: list[torch.Generator] | None):
    """Inside the block, every SeededDropout in `net` uses `generators` (None = plain dropout)."""
    layers = [m for m in net.modules() if isinstance(m, SeededDropout)]
    for m in layers:
        m.generators = generators
    try:
        yield
    finally:
        for m in layers:
            m.generators = None


class UnetSkipBlock(nn.Module):
    """One U-Net level: down -> submodule -> up, with the input concatenated to the output (except outermost)."""

    def __init__(self, outer_nc: int, inner_nc: int, input_nc: int | None = None, submodule: nn.Module | None = None,
                 outermost: bool = False, innermost: bool = False, dropout: float = 0.0):
        super().__init__()
        self.outermost = outermost
        input_nc = outer_nc if input_nc is None else input_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=False)
        downrelu, downnorm = nn.LeakyReLU(0.2, True), nn.BatchNorm2d(inner_nc)
        uprelu, upnorm = nn.ReLU(True), nn.BatchNorm2d(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            layers = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=False)
            layers = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=False)
            layers = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if dropout > 0:
                layers.append(SeededDropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, cfg: GanNetConfig):
        super().__init__()
        ngf = cfg.ngf
        block = UnetSkipBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(cfg.num_downs - 5):
            block = UnetSkipBlock(ngf * 8, ngf * 8, submodule=block, dropout=cfg.dropout)
        block = UnetSkipBlock(ngf * 4, ngf * 8, submodule=block)
        block = UnetSkipBlock(ngf * 2, ngf * 4, submodule=block)
        block = UnetSkipBlock(ngf, ngf * 2, submodule=block)
        self.model = UnetSkipBlock(IMAGE_CHANNELS, ngf, input_nc=SKETCH_CHANNELS + IMAGE_CHANNELS, submodule=block,
                                   outermost=True)

    def forward(self, sketch: Tensor, reference: Tensor) -> Tensor:
        return self.model(torch.cat([sketch, reference], 1))


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN on [S; Y]: one raw score per patch (LSGAN, so no sigmoid)."""

    def __init__(self, cfg: GanNetConfig):
        super().__init__()
        ndf = cfg.ndf
        layers = [nn.Conv2d(SKETCH_CHANNELS + IMAGE_CHANNELS, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, cfg.n_layers_d + 1):
            prev, mult = mult, min(2**n, 8)
            stride = 2 if n < cfg.n_layers_d else 1
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride, 1, bias=False), nn.BatchNorm2d(ndf * mult),
                       nn.LeakyReLU(0.2, True)]
        layers.append(nn.Conv2d(ndf * mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, sketch: Tensor, image: Tensor) -> Tensor:
        return self.model(torch.cat([sketch, image], 1))


def init_weights(net: nn.Module, gain: float = 0.02) -> nn.Module:
    """pix2pix init: conv weights ~ N(0, 0.02), BatchNorm weights ~ N(1, 0.02), biases 0."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.zeros_(m.bias)
    return net


def build_generator(cfg: GanNetConfig) -> UnetGenerator:
    return init_weights(UnetGenerator(cfg))


def build_discriminator(cfg: GanNetConfig) -> PatchDiscriminator:
    return init_weights(PatchDiscriminator(cfg))


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
