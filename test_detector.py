"""Standalone test for detector.py (no pytest; run with python)."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from detector import gradcam, load_detector, predict

CKPT = 'weights/pre-train/efficientnet/stargan.pth.tar'
CPU = torch.device('cpu')

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT), reason="bundled weights not present (e.g. CI)"
)


def _rand_image(n=64):
    return Image.fromarray(np.random.RandomState(0).randint(0, 256, (n, n, 3), dtype=np.uint8))


def test_load_and_forward():
    model = load_detector(CKPT, device=CPU)
    assert model.training is False, "model should be in eval() mode"
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == (1, 2), f"expected (1,2), got {tuple(out.shape)}"


def test_predict_sums_to_one():
    model = load_detector(CKPT, device=CPU)
    p = predict(model, _rand_image(), device=CPU, size=64)
    assert set(p) == {'Real', 'Fake'}
    assert abs(p['Real'] + p['Fake'] - 1.0) < 1e-4
    assert all(np.isfinite(v) for v in p.values())


def test_gradcam_overlay():
    model = load_detector(CKPT, device=CPU)
    ov = gradcam(model, _rand_image(), device=CPU, size=64)
    assert isinstance(ov, Image.Image)
    assert ov.size == (64, 64)
    assert np.isfinite(np.asarray(ov, dtype=np.float32)).all()
    # hooks must be removed after the call. Full backward hooks live in
    # _backward_hooks on torch 2.12 (and _full_backward_hooks on some builds);
    # check whichever exists.
    assert len(model._conv_head._forward_hooks) == 0
    bwd = (getattr(model._conv_head, '_full_backward_hooks', None)
           or getattr(model._conv_head, '_backward_hooks', None) or {})
    assert len(bwd) == 0


if __name__ == '__main__':
    test_load_and_forward()
    test_predict_sums_to_one()
    test_gradcam_overlay()
    print("detector load/predict/gradcam tests passed.")
