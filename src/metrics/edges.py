"""Edge-map F1 between an output and its sketch: guards the diversity metric.

A model can look "diverse" by ignoring the line art; this checks the output's edges still sit on the
sketch's lines. Both images go through the same Canny detector, and an edge pixel counts as matched
if an edge in the other map lies within `tolerance` pixels.
"""

import cv2
import numpy as np


def canny(gray_uint8: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv2.Canny(gray_uint8, low, high) > 0


def _distance_to(edges: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the nearest edge pixel."""
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)


def edge_f1(pred_rgb: np.ndarray, sketch_gray: np.ndarray, tolerance: float = 2.0) -> tuple[float, float, float]:
    """pred_rgb: (H,W,3) uint8, sketch_gray: (H,W) uint8 -> (precision, recall, f1)."""
    pred_edges = canny(cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY))
    sketch_edges = canny(sketch_gray)
    if not pred_edges.any() or not sketch_edges.any():
        return 0.0, 0.0, 0.0
    precision = float((_distance_to(sketch_edges)[pred_edges] <= tolerance).mean())
    recall = float((_distance_to(pred_edges)[sketch_edges] <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
