"""Standalone test for detector.py (no pytest; run with python)."""
import numpy as np
import torch
from PIL import Image

from detector import load_detector, predict

CKPT = 'weights/pre-train/efficientnet/stargan.pth.tar'
CPU = torch.device('cpu')


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


if __name__ == '__main__':
    test_load_and_forward()
    test_predict_sums_to_one()
    print("detector load/predict tests passed.")
