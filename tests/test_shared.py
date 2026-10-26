"""Checkpointing, EMA, CSV logging and the shared sampling harness."""

import sys

import pandas as pd
import pytest
import torch
import yaml
from torch import nn

import scripts.sample as sample
from src.colorizer import Colorizer
from src.utils.checkpoint import CheckpointManager, list_checkpoints
from src.utils.ema import EMA
from src.utils.logging import CSVLogger


def test_checkpoint_roundtrip_prune_and_rng(tmp_path):
    model = nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(4, 3)).sum().backward()
    opt.step()
    ckpt = CheckpointManager(tmp_path / "ck", keep=2)
    for step in (10, 20, 30):
        ckpt.save(step, model=model, optimizer=opt, data={"epoch": 1, "start": step})
    assert [s for s, _ in list_checkpoints(tmp_path / "ck")] == [20, 30]
    expected_next = torch.rand(1)

    model2 = nn.Linear(3, 2)
    opt2 = torch.optim.Adam(model2.parameters())
    torch.rand(5)  # advance the RNG; load_latest must restore it
    state = CheckpointManager(tmp_path / "ck").load_latest(model=model2, optimizer=opt2)
    assert state["step"] == 30 and state["data"] == {"epoch": 1, "start": 30}
    assert torch.equal(model2.weight, model.weight)
    assert opt2.state_dict()["state"][0]["step"] == 1
    assert torch.equal(torch.rand(1), expected_next)


def test_checkpoint_init_from_previous_session(tmp_path):
    model = nn.Linear(2, 2)
    CheckpointManager(tmp_path / "old").save(5, model=model)
    fresh = CheckpointManager(tmp_path / "new", init_from=tmp_path / "old")
    other = nn.Linear(2, 2)
    assert fresh.load_latest(model=other)["step"] == 5 and torch.equal(other.weight, model.weight)
    assert CheckpointManager(tmp_path / "empty").load_latest(model=other) is None


def test_ema_tracks_model():
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = EMA(model, decay=0.5, warmup=False)
    model.weight.data.fill_(1.0)
    ema.update(model)
    assert ema.model.weight.item() == pytest.approx(0.5)
    restored = EMA(model, decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.model.weight.item() == pytest.approx(0.5) and restored.num_updates == 1


def test_csv_logger_resumes(tmp_path):
    path = tmp_path / "log.csv"
    CSVLogger(path, echo=False).log(1, loss=0.5)
    log = CSVLogger(path, echo=False)
    log.log(2, loss=0.4)
    with pytest.raises(KeyError):
        log.log(3, lsos=0.3)
    df = pd.read_csv(path)
    assert list(df["step"]) == [1, 2] and list(df.columns) == ["step", "loss", "elapsed_min"]


class InvertColorizer(Colorizer):
    """Deterministic toy model: returns the reference shifted by the seed."""

    name = "toy"
    calls = 0

    def __init__(self, shift: float = 0.0, steps: int = 4):
        self.shift, self.steps = shift, steps

    def colorize(self, sketch, reference, seed, steps=None):
        InvertColorizer.calls += 1
        return (reference * -1 + self.shift + 0.01 * seed).clamp(-1, 1)

    def nfe(self, steps=None):
        return 2 * (steps or self.steps)


def run_sample(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sample", *argv])
    sample.main()


def test_sample_writes_contract_and_resumes(tmp_path, monkeypatch):
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({"name": "toy", "colorizer": "tests.test_shared:InvertColorizer",
                                   "model": {"shift": 0.1, "steps": 4}}))
    common = ["--config", str(cfg), "--dataset", "tests.fake_data:build_dataset", "--seeds", "0", "1",
              "--batch-size", "3", "--num-workers", "0", "--outputs-dir", str(tmp_path / "out"),
              "--results-dir", str(tmp_path / "res"), "--device", "cpu"]
    run_sample(monkeypatch, *common, "--mode", "self_ref", "cross_ref")
    for mode in ("self_ref", "cross_ref"):
        split = "test_cross" if mode == "cross_ref" else "test"  # each mode's own test split
        names = sorted(p.name for p in (tmp_path / "out/toy" / mode).glob("*.png"))
        assert names == sorted(f"{split}{i:03d}_s{s}.png" for i in range(4) for s in (0, 1))
    speed = pd.read_csv(tmp_path / "res/speed.csv")
    assert list(speed.columns) == sample.SPEED_COLUMNS and len(speed) == 2
    assert speed.iloc[0]["nfe"] == 8 and speed.iloc[0]["steps"] == 4

    calls = InvertColorizer.calls
    run_sample(monkeypatch, *common, "--mode", "self_ref")
    assert InvertColorizer.calls == calls  # everything existed -> nothing regenerated
    assert len(pd.read_csv(tmp_path / "res/speed.csv")) == 2

    run_sample(monkeypatch, *common, "--mode", "self_ref", "--steps", "2", "--set", "shift=0.0")
    assert (tmp_path / "out/toy_steps2/self_ref/test000_s1.png").exists()
    assert pd.read_csv(tmp_path / "res/speed.csv").iloc[-1].to_dict()["model"] == "toy_steps2"


def test_resolve_colorizer_by_prefix():
    with pytest.raises(ValueError, match="cannot pick"):
        sample.resolve_colorizer("mystery", None)
    assert sample.resolve_colorizer("controlnet_ip0.6", None).__name__ == "ControlNetColorizer"
    assert sample.parse_overrides(["a=0.5", "b=null", "c=text"]) == {"a": 0.5, "b": None, "c": "text"}
