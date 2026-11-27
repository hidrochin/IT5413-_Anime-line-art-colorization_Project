"""The single source of evaluation seeds (fairness contract rule 2)."""

import random

import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 4]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
