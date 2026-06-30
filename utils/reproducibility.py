"""Reproducibility helpers: seed every RNG the pipeline touches."""
import os
import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices); put cuDNN in
    deterministic mode.

    Determinism is favored over throughput — this is a portfolio repo where
    reproducible numbers matter more than peak speed. Call once, immediately
    after options are parsed and before building any model, dataset, or loader.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
