"""Create a tiny synthetic ImageFolder dataset for CPU smoke-testing the
training/eval entry points. Layout:

dataset/_smoke/
    train/{fake,real}/             8 images each
    val/{fake,real}/               8 images each
    test/{fake,real}/              8 images each
    style2/2000_shot/{fake,real}/  8 images each  (transfer.py train dir)

Also pre-creates the log/smoke* checkpoint dirs the smoke runs write to.
Run from the repo root: python scripts/make_smoke_dataset.py
"""
import os

import numpy as np
from PIL import Image

ROOT = os.path.join('dataset', '_smoke')
SPLITS = ['train', 'val', 'test', os.path.join('style2', '2000_shot')]
CLASSES = ['fake', 'real']
N_IMAGES = 8
SIZE = 64

rng = np.random.default_rng(7)

for split in SPLITS:
    for cls in CLASSES:
        d = os.path.join(ROOT, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(N_IMAGES):
            arr = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(d, f'{cls}_{i}.png'))

for sub in ['smoke', 'smoke_transfer', 'smoke_pretrain', 'smoke_es']:
    os.makedirs(os.path.join('log', sub), exist_ok=True)

print(f'Smoke dataset written to {ROOT}')
