"""Standalone test for utils.reproducibility (no pytest; run with python)."""
import random

import numpy as np
import torch

from utils.reproducibility import set_seeds


def test_torch_repeatable():
    set_seeds(7)
    a = torch.rand(5)
    set_seeds(7)
    b = torch.rand(5)
    assert torch.equal(a, b), "torch.rand not reproducible after set_seeds"


def test_numpy_and_random_repeatable():
    set_seeds(123)
    a_np, a_py = np.random.rand(5).tolist(), [random.random() for _ in range(5)]
    set_seeds(123)
    b_np, b_py = np.random.rand(5).tolist(), [random.random() for _ in range(5)]
    assert a_np == b_np, "numpy not reproducible"
    assert a_py == b_py, "random not reproducible"


def test_cudnn_deterministic_flag():
    set_seeds(1)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


if __name__ == '__main__':
    test_torch_repeatable()
    test_numpy_and_random_repeatable()
    test_cudnn_deterministic_flag()
    print("All reproducibility tests passed.")
