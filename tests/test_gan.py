"""Model A wiring with a tiny network, a fake VGG and a fake LPIPS (no weights are downloaded)."""

import sys

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

import scripts.sample as sample
import scripts.train_gan as train_gan
from src.models.gan import (GanColorizer, GanConfig, GanNetConfig, build_discriminator, build_generator,
                            seeded_dropout)
from src.models.gan.losses import VGGPerceptualLoss, lsgan_d_loss, lsgan_g_loss
from tests.fake_data import FakeLPIPS

TINY = {"ngf": 4, "ndf": 4}


def fake_vgg() -> VGGPerceptualLoss:
    """30 layers like vgg19().features[:30], 3 channels throughout."""
    layers = [nn.Conv2d(3, 3, 3, padding=1) if i % 2 == 0 else nn.ReLU() for i in range(30)]
    return VGGPerceptualLoss(features=nn.Sequential(*layers))


def make_colorizer(**cfg) -> GanColorizer:
    torch.manual_seed(0)
    return GanColorizer(GanConfig(device="cpu", **cfg), net=build_generator(GanNetConfig.from_dict(TINY)))


def test_shapes_and_patch_size():
    cfg = GanNetConfig()
    g, d = build_generator(cfg), build_discriminator(cfg)
    assert 50e6 < sum(p.numel() for p in g.parameters()) < 60e6  # pix2pix U-Net-256: 54.4M
    s, r = torch.zeros(2, 1, 256, 256), torch.zeros(2, 3, 256, 256)
    assert g(s, r).shape == (2, 3, 256, 256)
    assert d(s, r).shape == (2, 1, 30, 30)  # 70x70 PatchGAN on 256 px
    assert sum(isinstance(m, nn.Dropout) for m in g.modules()) == 3


def test_losses():
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    vgg = fake_vgg()
    assert vgg(x, x).item() == 0 and vgg(x, -x).item() > 0
    assert lsgan_d_loss(torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)).item() == 0
    assert lsgan_g_loss(torch.ones(2, 1, 4, 4)).item() == 0


def test_colorize_contract_and_seeds():
    colorizer = make_colorizer()
    sketch, ref = torch.zeros(3, 1, 256, 256), torch.rand(3, 3, 256, 256) * 2 - 1
    sketch[:] = sketch[0]
    ref[:] = ref[0]
    out = colorizer.colorize(sketch, ref, seed=1)
    assert out.shape == (3, 3, 256, 256) and out.min() >= -1 and out.max() <= 1
    assert torch.equal(out, colorizer.colorize(sketch, ref, seed=1))
    assert torch.equal(out[0], out[2])  # same inputs + same seed -> same output, whatever the batch slot
    assert torch.equal(out[:1], colorizer.colorize(sketch[:1], ref[:1], seed=1))  # ... and whatever the batch size
    assert not torch.equal(out, colorizer.colorize(sketch, ref, seed=2))  # dropout gives each seed its own output
    assert colorizer.nfe() == 1

    plain = make_colorizer(test_dropout=False)
    assert torch.equal(plain.colorize(sketch, ref, seed=1), plain.colorize(sketch, ref, seed=2))


def test_seeded_dropout_is_restored():
    g = build_generator(GanNetConfig.from_dict(TINY)).eval()
    x, r = torch.zeros(1, 1, 256, 256), torch.zeros(1, 3, 256, 256)
    with pytest.raises(ValueError, match="generators"):
        with seeded_dropout(g, [torch.Generator(), torch.Generator()]):
            g(x, r)
    with torch.no_grad():
        assert torch.equal(g(x, r), g(x, r))  # back to plain (eval -> identity) dropout


def test_sample_resolves_gan_runs():
    for run in ("gan", "gan_nodrop"):
        assert sample.resolve_colorizer(run, None) is GanColorizer
    with pytest.raises(ValueError, match="checkpoint"):
        GanColorizer(GanConfig(device="cpu"))


def run_train(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(train_gan, "VGGPerceptualLoss", fake_vgg)
    monkeypatch.setattr(train_gan, "LPIPS", FakeLPIPS)
    monkeypatch.setattr(sys, "argv", ["train_gan", "--config", str(tmp_path / "gan.yaml"), "--device", "cpu",
                                      "--dataset", "tests.fake_data:build_dataset", *extra])
    train_gan.main()


def write_config(tmp_path, run="run", **train):
    out = tmp_path / run
    (tmp_path / "gan.yaml").write_text(yaml.safe_dump({
        "name": "gan", "arch": TINY,
        "train": {"output_dir": str(out), "batch_size": 2, "max_steps": 4, "lr_decay_start": 2, "num_workers": 0,
                  "log_every": 1, "val_every": 2, "num_val": 3, "val_batch": 2, **train},
    }))
    return out


def test_train_resume_and_load(tmp_path, monkeypatch):
    out = write_config(tmp_path)
    run_train(monkeypatch, tmp_path)
    log = pd.read_csv(out / "log.csv")
    assert list(log["step"]) == [1, 2, 3, 4] and log["loss_l1"].notna().all()
    assert log["val_lpips"].notna().sum() == 2
    assert list(log["lr"]) == pytest.approx([2e-4, 2e-4, 2e-4, 1e-4])  # constant, then linear decay from step 2
    assert (out / "samples/step_00000004.png").exists()
    assert (out / "gan_G.pt").exists() and (out / "gan_best.pt").exists()

    run_train(monkeypatch, tmp_path, "--max-steps", "6")  # resumes from step 4
    assert list(pd.read_csv(out / "log.csv")["step"]) == [1, 2, 3, 4, 5, 6]
    assert torch.load(out / "gan_G.pt", weights_only=False)["step"] == 6

    for path, step in ((out / "gan_G.pt", 6), (out / "checkpoints", None), (out, None)):
        colorizer = GanColorizer(GanConfig(checkpoint=str(path), device="cpu"))
        assert step is None or colorizer.step == step
        assert colorizer.colorize(torch.zeros(1, 1, 256, 256), torch.zeros(1, 3, 256, 256), seed=0).shape == (1, 3, 256, 256)


def test_train_cpu_warp_and_no_warp_paths(tmp_path, monkeypatch):
    for run, train in (("cpu_warp", {"gpu_warp": False}), ("no_warp", {"reference_aug": False})):
        out = write_config(tmp_path, run, max_steps=1, **train)
        run_train(monkeypatch, tmp_path)
        assert (out / "gan_G.pt").exists()
