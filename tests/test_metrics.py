import math

import numpy as np
import pytest
import torch

from src.metrics import check_metrics
from src.metrics.distribution import frechet_distance, kernel_distance
from src.metrics.edges import edge_f1
from src.metrics.files import parse_output_name
from src.metrics.perceptual import pairwise_diversity
from src.metrics.pixel import ms_ssim, psnr, ssim
from src.metrics.reference import color_distance
from tests.fake_data import FakeLPIPS, make_pair


def test_parse_output_name():
    assert parse_output_name("abc_s3.png") == ("abc", 3)
    assert parse_output_name("a_b_s12.png") == ("a_b", 12)
    assert parse_output_name("gt_image.png") == ("gt_image", 0)


def test_psnr_known_value():
    target = torch.zeros(2, 3, 8, 8)
    assert psnr(target + 0.5, target)[0].item() == pytest.approx(10 * math.log10(4), rel=1e-5)
    assert psnr(target, target)[0].item() == pytest.approx(100.0)


def test_ssim_and_ms_ssim_identical():
    x = torch.rand(2, 3, 256, 256)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    assert torch.allclose(ms_ssim(x, x), torch.ones(2), atol=1e-4)
    assert ssim(x, torch.rand_like(x)).mean() < 0.5


def test_pairwise_diversity():
    same = torch.rand(1, 3, 16, 16).repeat(5, 1, 1, 1)
    assert pairwise_diversity(same, FakeLPIPS()) == 0.0
    assert pairwise_diversity(torch.rand(5, 3, 16, 16), FakeLPIPS()) > 0.1
    assert math.isnan(pairwise_diversity(same[:1], FakeLPIPS()))


def test_edge_f1():
    color, sketch = make_pair(0)
    _, _, f1 = edge_f1(color, sketch)
    assert f1 > 0.8
    blank = np.full_like(color, 128)
    assert edge_f1(blank, sketch) == (0.0, 0.0, 0.0)
    _, _, f1_other = edge_f1(make_pair(7)[0], sketch)
    assert f1_other < f1


def test_color_distance():
    color, _ = make_pair(0)
    assert color_distance(color, color) == pytest.approx(0.0, abs=1e-6)
    red = np.zeros((64, 64, 3), np.uint8)
    red[..., 0] = 255
    blue = np.zeros((64, 64, 3), np.uint8)
    blue[..., 2] = 255
    assert color_distance(red, blue, use_mask=False) > 0.9


def test_frechet_and_kernel_distance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    b = rng.normal(size=(500, 8))
    assert frechet_distance(a, b) < 0.2
    assert frechet_distance(a, b + 2.0) == pytest.approx(8 * 4.0, rel=0.1)
    assert abs(kernel_distance(a, b, num_subsets=10, max_subset_size=200)) < 0.05
    assert kernel_distance(a, b + 2.0, num_subsets=10, max_subset_size=200) > 0.5


def test_mode_rules():
    check_metrics("self_ref", ["psnr", "fid", "diversity"])
    with pytest.raises(ValueError, match="not valid in mode 'cross_ref'"):
        check_metrics("cross_ref", ["psnr"])
    with pytest.raises(ValueError):
        check_metrics("self_ref", ["color_dist"])
    with pytest.raises(ValueError, match="unknown mode"):
        check_metrics("train", ["fid"])
