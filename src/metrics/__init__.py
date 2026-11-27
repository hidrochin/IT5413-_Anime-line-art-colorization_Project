"""Metric library used by scripts/evaluate.py and scripts/tune_controlnet.py.

Which metric is valid in which evaluation mode is defined once, here (CLAUDE.md "Experiments").
"""

PAIRED = ["psnr", "ssim", "ms_ssim", "lpips"]  # need ground truth -> self_ref only
DISTRIBUTION = ["fid", "kid", "fd_dinov2"]
ALWAYS = ["diversity", "edge_f1"]
REFERENCE = ["color_dist", "clip_sim"]  # need a real (non-warped-GT) reference -> cross_ref only

MODE_METRICS = {
    "self_ref": PAIRED + DISTRIBUTION + ALWAYS,
    "cross_ref": DISTRIBUTION + ALWAYS + REFERENCE,
    "shuffled_ref": DISTRIBUTION + ALWAYS,
}


def check_metrics(mode: str, metrics: list[str]) -> None:
    if mode not in MODE_METRICS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODE_METRICS)}")
    invalid = [m for m in metrics if m not in MODE_METRICS[mode]]
    if invalid:
        raise ValueError(f"metrics {invalid} are not valid in mode {mode!r}; allowed: {MODE_METRICS[mode]}")
