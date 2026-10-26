"""Tuning search, LoRA sampler, export, tables, plots and user study on synthetic data."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

import scripts.tune_controlnet as tune
from scripts.train_lora import EpochSampler
from tests.fake_data import FakeLPIPS


def run_main(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_epoch_sampler_resumes_same_order():
    full = list(EpochSampler(10, seed=3))
    s = EpochSampler(10, seed=3)
    s.set_position(0, 4)
    assert list(s) == full[4:] and len(s) == 6
    s.set_position(1, 0)
    assert list(s) != full and sorted(s) == list(range(10))


class FakeColorizer:
    """Output quality peaks at ip_adapter_scale=0.8 and controlnet_scale=1.0."""

    built = 0

    def __init__(self, cfg):
        FakeColorizer.built += 1
        self.cfg = cfg

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self.cfg, k, v)

    def colorize(self, sketch, reference, seed, steps=None):
        err = abs(self.cfg.ip_adapter_scale - 0.8) + abs(self.cfg.controlnet_scale - 1.0)
        return (torch.ones(sketch.shape[0], 3, 256, 256) * (1 - err)).clamp(-1, 1)


def test_tune_coordinate_search_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "ControlNetColorizer", FakeColorizer)
    monkeypatch.setattr(tune, "LPIPS", FakeLPIPS)
    monkeypatch.setattr(tune, "load_val", lambda spec, split, n: [
        {"sketch": torch.zeros(1, 256, 256), "reference": torch.zeros(3, 256, 256), "target": torch.ones(3, 256, 256)}
    ] * n)
    cfg = yaml.safe_load(Path("configs/controlnet.yaml").read_text())
    cfg["model"]["device"] = "cpu"
    cfg["tune"].update(num_images=2, batch_size=2, order=["ip_adapter_scale", "controlnet_scale"])
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "results/tune.csv"

    run_main(monkeypatch, tune, "--config", str(cfg_path), "--out", str(out))
    best = yaml.safe_load((tmp_path / "results/tune_controlnet_best.yaml").read_text())["model"]
    assert best["ip_adapter_scale"] == 0.8 and best["controlnet_scale"] == 1.0
    n_rows = len(pd.read_csv(out))
    assert n_rows == len(cfg["tune"]["grid"]["ip_adapter_scale"]) + len(cfg["tune"]["grid"]["controlnet_scale"]) - 1
    assert FakeColorizer.built == 1  # base model never changed -> one pipeline

    run_main(monkeypatch, tune, "--config", str(cfg_path), "--out", str(out))
    assert len(pd.read_csv(out)) == n_rows  # second run is fully resumed from the CSV


def test_export_eval_assets(tmp_path, monkeypatch):
    import scripts.export_eval_assets as export

    run_main(monkeypatch, export, "--out", str(tmp_path), "--dataset", "tests.fake_data:build_dataset")
    for sub in ("ref_self_ref", "ref_cross_ref", "ref_shuffled_ref"):
        assert len(list((tmp_path / sub).glob("*.png"))) == 4
    # gt/ and sketch/ hold both test sets: test (self/shuffled) and test_cross (cross_ref)
    for sub in ("gt", "sketch"):
        assert len(list((tmp_path / sub).glob("*.png"))) == 8
    assert {p.stem for p in (tmp_path / "ref_cross_ref").glob("*.png")} == {f"test_cross{i:03d}" for i in range(4)}


def write_results(results: Path):
    rows = [
        ("gan", 0.30, 0.01, 2), ("flow_w1.0", 0.25, 0.20, 1), ("flow_w3.0", 0.20, 0.10, 1),
        ("flow_steps4", 0.28, 0.15, 4), ("flow_steps16", 0.22, 0.12, 16), ("controlnet", 0.35, 0.25, 20),
    ]
    for name, lp, div, steps in rows:
        pd.DataFrame([{"timestamp": "2026-11-01T00:00:00", "model": name, "mode": "self_ref", "psnr": 20.0,
                       "ms_ssim": 0.9, "lpips": lp, "diversity": div, "fd_dinov2": 300 * lp}]).to_csv(
            results / f"{name}_self_ref.csv", index=False)
    pd.DataFrame([{"model": "flow_steps4", "steps": 4, "nfe": 8, "sec_per_image": 0.05, "gpu": "T4"},
                  {"model": "flow_steps16", "steps": 16, "nfe": 32, "sec_per_image": 0.2, "gpu": "T4"},
                  {"model": "gan", "steps": 1, "nfe": 1, "sec_per_image": 0.01, "gpu": "T4"}]).to_csv(results / "speed.csv", index=False)


def test_tables_and_plots(tmp_path, monkeypatch):
    import scripts.make_results_table as table
    import scripts.plot_diversity_fidelity as divfid
    import scripts.plot_quality_vs_steps as steps

    write_results(tmp_path)
    run_main(monkeypatch, table, "--mode", "self_ref", "--models", "gan", "controlnet", "--results-dir", str(tmp_path))
    md = (tmp_path / "table_self_ref.md").read_text()
    assert "AnimeDiffusion" in md and "gan" in md
    run_main(monkeypatch, divfid, "--series", "gan", "flow_w*", "--results-dir", str(tmp_path), "--out", str(tmp_path / "a.png"))
    run_main(monkeypatch, steps, "--series", "flow_steps*", "--results-dir", str(tmp_path), "--out", str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "b.png").stat().st_size > 0


def test_plot_samples(eval_tree, monkeypatch):
    import scripts.plot_samples as samples

    common = ["--models", "good", "noisy", "--eval-dir", str(eval_tree / "data/eval"), "--outputs-dir", str(eval_tree / "outputs")]
    run_main(monkeypatch, samples, *common, "--n", "2", "--out", str(eval_tree / "s.png"))
    run_main(monkeypatch, samples, *common, "--seeds-for", "img001", "--out", str(eval_tree / "d.png"))
    assert (eval_tree / "s.png").exists() and (eval_tree / "d.png").exists()


def test_user_study_build_and_analyze(eval_tree, monkeypatch):
    import scripts.analyze_user_study as analyze
    import scripts.build_user_study as build

    study = eval_tree / "study"
    run_main(monkeypatch, build, "--models", "good", "noisy", "other", "--n-cases", "3",
             "--eval-dir", str(eval_tree / "data/eval"), "--outputs-dir", str(eval_tree / "outputs"), "--out", str(study))
    page = (study / "study.html").read_text()
    for model in ("good", "noisy", "other"):
        assert f">{model}<" not in page and f'"{model}"' not in page  # blind: no model names in the form
    key = pd.read_csv(study / "key.csv")
    assert len(key) == 9 and set(key["model"]) == {"good", "noisy", "other"}

    rank_of = {"good": 1, "noisy": 2, "other": 3}
    rows = [{"rater": r, "case": k.case, "criterion": c, "option": k.option, "rank": rank_of[k.model]}
            for r in ("r1", "r2") for k in key.itertuples() for c in ("color_match", "overall")]
    pd.DataFrame(rows).to_csv(study / "responses/all.csv", index=False)
    run_main(monkeypatch, analyze, "--study-dir", str(study), "--out", str(eval_tree / "summary.csv"))
    summary = pd.read_csv(eval_tree / "summary.csv")
    good = summary[(summary.model == "good") & (summary.criterion == "overall")].iloc[0]
    assert good["mean_rank"] == 1.0 and good["first_place"] == 1.0
    assert summary["friedman_p"].max() < 0.05


def test_user_study_needs_enough_cases(eval_tree, monkeypatch):
    import scripts.build_user_study as build

    with pytest.raises(SystemExit, match="need 30"):
        run_main(monkeypatch, build, "--models", "good", "noisy", "--eval-dir", str(eval_tree / "data/eval"),
                 "--outputs-dir", str(eval_tree / "outputs"), "--out", str(eval_tree / "study"))
