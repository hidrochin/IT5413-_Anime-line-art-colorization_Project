"""Model B wiring with a fake VAE and a fake DINOv2 (no weights are downloaded)."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

import scripts.sample as sample
import scripts.train_flow as train_flow
from src.models.flow import FlowColorizer, FlowConfig, FlowNetConfig, ReferenceEncoder, build_flow_net, euler_sample
from src.models.flow.model import flow_matching_loss, num_ref_tokens, sample_t

TINY = {"block_out_channels": [32, 32, 32, 32], "layers_per_block": 1, "head_dim": 32}


class FakeVAE(nn.Module):
    """8x average-pool 'encoder' and nearest-upsample 'decoder' with the diffusers call shapes."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=0.18215)

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def encode(self, x):
        z = F.avg_pool2d(x, 8)
        z = torch.cat([z, z.mean(1, keepdim=True)], 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=z, sample=lambda: z))

    def decode(self, z):
        return SimpleNamespace(sample=F.interpolate(z[:, :3], scale_factor=8, mode="nearest"))


class FakeDino(nn.Module):
    """224x224 -> 1 CLS + 16x16 patch tokens of width 384, like dinov2-small."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 384)

    def forward(self, pixel_values):
        p = F.avg_pool2d(pixel_values, 14).flatten(2).transpose(1, 2)  # (B, 256, 3)
        tokens = self.proj(p)
        return SimpleNamespace(last_hidden_state=torch.cat([tokens.mean(1, keepdim=True), tokens], 1))


def fake_ref_encoder(kind="dinov2"):
    return ReferenceEncoder(kind, model=FakeDino())


def make_colorizer(**cfg):
    net = build_flow_net(FlowNetConfig.from_dict(TINY))
    return FlowColorizer(FlowConfig(device="cpu", **cfg), net=net, vae=FakeVAE(), ref_encoder=fake_ref_encoder())


def test_reference_tokens_shape():
    tokens = fake_ref_encoder()(torch.zeros(2, 3, 256, 256))
    assert tokens.shape == (2, num_ref_tokens("dinov2"), 384) == (2, 256, 384)


def test_flow_loss_and_euler_shapes():
    net = build_flow_net(FlowNetConfig.from_dict(TINY))
    z = torch.randn(3, 4, 32, 32)
    tokens = torch.randn(3, 256, 384)
    t = sample_t(3, "cpu")
    assert ((t > 0) & (t < 1)).all()
    loss = flow_matching_loss(net, z, z, tokens, t)
    loss.backward()
    assert torch.isfinite(loss) and net.null_tokens.grad is None  # null tokens only train via CFG dropout
    assert euler_sample(net, z, z, tokens, steps=3, guidance=2.0).shape == z.shape


def test_euler_is_exact_for_a_straight_path():
    """With the true velocity of a straight path, 1 Euler step lands exactly on the data."""

    class Oracle(nn.Module):
        def __init__(self, z1, eps):
            super().__init__()
            self.v = z1 - eps

        def null_cond(self, b):
            return torch.zeros(b, 1, 1)

        def forward(self, x, t, zs, tokens):
            return self.v.repeat(x.shape[0] // self.v.shape[0], 1, 1, 1)

    z1, eps = torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32)
    for steps, w in ((1, 1.0), (4, 3.0)):
        out = euler_sample(Oracle(z1, eps), eps, z1, torch.zeros(2, 1, 1), steps, w)
        assert torch.allclose(out, z1, atol=1e-5)


def test_colorize_contract_and_seeds():
    colorizer = make_colorizer(steps=2, guidance_scale=2.0)
    sketch, ref = torch.zeros(3, 1, 256, 256), torch.zeros(3, 3, 256, 256)
    out = colorizer.colorize(sketch, ref, seed=1)
    assert out.shape == (3, 3, 256, 256) and out.min() >= -1 and out.max() <= 1
    assert torch.equal(out, colorizer.colorize(sketch, ref, seed=1))
    assert torch.equal(out[0], out[1])  # same inputs + same seed -> same output, whatever the batch slot
    assert not torch.equal(out, colorizer.colorize(sketch, ref, seed=2))
    assert colorizer.nfe() == 4 and colorizer.nfe(8) == 16
    assert make_colorizer(guidance_scale=1.0).nfe(8) == 8


def test_sample_resolves_flow_runs():
    for run in ("flow", "flow_steps8", "flow_w3.0"):
        assert sample.resolve_colorizer(run, None) is FlowColorizer
    with pytest.raises(ValueError, match="checkpoint"):
        FlowColorizer(FlowConfig(device="cpu"), vae=FakeVAE(), ref_encoder=fake_ref_encoder())


def run_train(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(train_flow, "load_vae", lambda *a, **k: FakeVAE())
    monkeypatch.setattr(train_flow, "ReferenceEncoder", lambda kind: fake_ref_encoder(kind))
    monkeypatch.setattr(sys, "argv", ["train_flow", "--config", str(tmp_path / "flow.yaml"), "--device", "cpu",
                                      "--dataset", "tests.fake_data:build_dataset", *extra])
    train_flow.main()


def test_train_resume_and_load(tmp_path, monkeypatch):
    from tests.fake_data import build_dataset

    # Latent cache in the cache_latents.py format, in a different order than the dataset.
    vae = FakeVAE()
    items = build_dataset("train", "self_ref")[::-1]
    from src.utils.vae import encode

    torch.save({"ids": [x["id"] for x in items],
                "sketch": encode(vae, torch.stack([x["sketch"] for x in items])).half(),
                "target": encode(vae, torch.stack([x["target"] for x in items])).half(),
                "vae": "fake", "scaling_factor": 0.18215}, tmp_path / "train.pt")
    out = tmp_path / "run"
    (tmp_path / "flow.yaml").write_text(yaml.safe_dump({
        "name": "flow", "arch": TINY,
        "train": {"output_dir": str(out), "latents_dir": str(tmp_path), "batch_size": 2, "max_steps": 4,
                  "warmup_steps": 2, "num_workers": 0, "log_every": 1, "sample_every": 2, "num_val": 2,
                  "val_steps": 2},
    }))

    run_train(monkeypatch, tmp_path)
    log = pd.read_csv(out / "log.csv")
    assert list(log["step"]) == [1, 2, 3, 4] and log["loss"].notna().all()
    assert log["val_loss"].notna().sum() == 2
    assert (out / "samples/step_00000004.png").exists() and (out / "flow_ema.pt").exists()

    run_train(monkeypatch, tmp_path, "--max-steps", "6")  # resumes from step 4
    assert list(pd.read_csv(out / "log.csv")["step"]) == [1, 2, 3, 4, 5, 6]
    assert torch.load(out / "flow_ema.pt", weights_only=False)["step"] == 6

    for path in (out / "flow_ema.pt", out / "checkpoints"):
        colorizer = FlowColorizer(FlowConfig(checkpoint=str(path), device="cpu", steps=2), vae=FakeVAE(),
                                  ref_encoder=fake_ref_encoder())
        assert colorizer.step == 6
        assert colorizer.colorize(torch.zeros(1, 1, 256, 256), torch.zeros(1, 3, 256, 256), seed=0).shape == (1, 3, 256, 256)


def test_train_rejects_missing_latents(tmp_path, monkeypatch):
    torch.save({"ids": ["other"], "sketch": torch.zeros(1, 4, 32, 32), "target": torch.zeros(1, 4, 32, 32)},
               tmp_path / "train.pt")
    (tmp_path / "flow.yaml").write_text(yaml.safe_dump({
        "arch": TINY, "train": {"output_dir": str(tmp_path / "run"), "latents_dir": str(tmp_path),
                                "batch_size": 2, "num_workers": 0}}))
    with pytest.raises(KeyError, match="no cached latents"):
        run_train(monkeypatch, tmp_path)
