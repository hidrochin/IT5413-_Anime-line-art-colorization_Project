import pandas as pd
import pytest

import scripts.evaluate as evaluate
from tests.fake_data import FakeLPIPS


@pytest.fixture(autouse=True)
def fake_lpips(monkeypatch):
    monkeypatch.setattr(evaluate, "LPIPS", FakeLPIPS)


def run(tree, *argv):
    return evaluate.evaluate(evaluate.parse_args([
        "--eval-dir", str(tree / "data/eval"), "--device", "cpu", *argv,
    ]))


def test_self_ref_paired_metrics(eval_tree):
    common = ["--mode", "self_ref", "--metrics", "psnr,ssim,ms_ssim,lpips,diversity,edge_f1"]
    good = run(eval_tree, "--model", "good", "--pred-dir", str(eval_tree / "outputs/good/self_ref"),
               "--out", str(eval_tree / "results/good_self_ref.csv"), *common)
    noisy = run(eval_tree, "--model", "noisy", "--pred-dir", str(eval_tree / "outputs/noisy/self_ref"),
                "--out", str(eval_tree / "results/noisy_self_ref.csv"), *common)
    assert good["n_ids"] == 4 and good["n_seeds"] == 3
    assert good["psnr"] == pytest.approx(100.0)
    assert good["lpips"] == pytest.approx(0.0)
    assert good["diversity"] == pytest.approx(0.0)
    assert noisy["psnr"] < 30 and noisy["diversity"] > 0
    assert good["edge_f1"] > noisy["edge_f1"]
    df = pd.read_csv(eval_tree / "results/good_self_ref.csv")
    assert len(df) == 1 and df.loc[0, "model"] == "good"


def test_rows_append(eval_tree):
    out = eval_tree / "results/good_self_ref.csv"
    for _ in range(2):
        run(eval_tree, "--model", "good", "--mode", "self_ref", "--metrics", "psnr",
            "--pred-dir", str(eval_tree / "outputs/good/self_ref"), "--out", str(out))
    assert len(pd.read_csv(out)) == 2


def test_cross_ref_color_distance(eval_tree):
    kw = ["--mode", "cross_ref", "--metrics", "color_dist"]
    good = run(eval_tree, "--model", "good", "--pred-dir", str(eval_tree / "outputs/good/cross_ref"),
               "--out", str(eval_tree / "r1.csv"), *kw)
    other = run(eval_tree, "--model", "other", "--pred-dir", str(eval_tree / "outputs/other/cross_ref"),
                "--out", str(eval_tree / "r2.csv"), *kw)
    assert good["color_dist"] == pytest.approx(0.0, abs=1e-6)
    assert other["color_dist"] > good["color_dist"]


def test_paired_metric_rejected_in_cross_ref(eval_tree):
    with pytest.raises(ValueError, match="not valid"):
        run(eval_tree, "--model", "good", "--mode", "cross_ref", "--metrics", "psnr",
            "--pred-dir", str(eval_tree / "outputs/good/cross_ref"))


def test_missing_outputs(eval_tree):
    (eval_tree / "outputs/good/self_ref/img000_s0.png").unlink()
    for s in (1, 2):
        (eval_tree / f"outputs/good/self_ref/img003_s{s}.png").unlink()
    (eval_tree / "outputs/good/self_ref/img003_s0.png").unlink()
    args = ["--model", "good", "--mode", "self_ref", "--metrics", "psnr",
            "--pred-dir", str(eval_tree / "outputs/good/self_ref"), "--out", str(eval_tree / "r.csv")]
    with pytest.raises(ValueError, match="no output"):
        run(eval_tree, *args)
    row = run(eval_tree, *args, "--allow-partial")
    assert row["n_ids"] == 3


def test_ground_truth_as_sanity_run(eval_tree):
    row = run(eval_tree, "--model", "gt", "--mode", "self_ref", "--metrics", "psnr,edge_f1,diversity",
              "--pred-dir", str(eval_tree / "data/eval/gt"), "--out", str(eval_tree / "gt.csv"))
    assert row["n_seeds"] == 1 and row["psnr"] == pytest.approx(100.0)
