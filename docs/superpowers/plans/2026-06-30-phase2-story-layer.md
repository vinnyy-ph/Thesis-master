# Phase 2: Story Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo reproducible and legible — deterministic seeding, a real metrics module (single-class-safe), machine-readable run export, a SynerMix-focused README, and pinned dependencies.

**Architecture:** Three new focused helpers under `utils/` (`reproducibility.py`, `metrics.py`, `run_logger.py`), each with a plain-`assert` test script (pytest is deferred to Phase 4 — it is not installed). The helpers are then wired into all six entry points: seeds at startup, end-of-epoch metrics via accumulation (which root-fixes the per-batch single-class `roc_auc_score` NaN), and per-run artifacts to `runs/<timestamp>/`. Finally the README is rewritten and `requirements.txt` pinned.

**Tech Stack:** Python 3.12 (`.\venv\Scripts\python.exe`), PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu, scikit-learn 1.8.0, numpy 2.4.4. Windows / PowerShell.

## Global Constraints

- ALWAYS prefix Python runs with `$env:PYTHONUTF8='1'; ` — scripts print emoji/✓ that crash Windows cp1252 console with `UnicodeEncodeError`.
- ALWAYS pass `--num_workers 0` to training/eval scripts — train transforms use `transforms.Lambda`, unpicklable by Windows DataLoader workers.
- This machine is CPU-only; the smoke dataset is `dataset/_smoke` (regenerate with `python scripts/make_smoke_dataset.py` if missing).
- Run everything from repo root: `C:\Users\Vincent\Documents\GitHub\Portfolio\Thesis-master`. Use `.\venv\Scripts\python.exe`.
- `dataset/`, `log/`, and (after Task 3) `runs/` are gitignored.
- pytest is NOT installed — Phase 2 tests are standalone scripts run with `python <test>.py`, exit 0 on pass. Do NOT add pytest (Phase 4).
- `compute_metrics` contract: `y_score` is the **positive-class probability in [0,1]** (i.e. `F.softmax(outputs, dim=1)[:, 1]`), NOT a raw logit. Thresholding at 0.5 then equals argmax over the two classes, matching existing top-1 accuracy.
- The canonical wiring pattern (Tasks 5–8) is defined once in Task 5; later wiring tasks reference and repeat it.

---

### Task 1: `utils/reproducibility.py` + test

**Files:**
- Create: `utils/reproducibility.py`
- Create: `test_reproducibility.py`

**Interfaces:**
- Produces: `set_seeds(seed: int) -> None`

- [ ] **Step 1: Write the test (it will fail — module doesn't exist yet)**

Create `test_reproducibility.py`:

```python
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
```

- [ ] **Step 2: Run it, confirm it fails on import**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_reproducibility.py
```
Expected: `ModuleNotFoundError: No module named 'utils.reproducibility'` (or ImportError for `set_seeds`).

- [ ] **Step 3: Implement `utils/reproducibility.py`**

Create `utils/reproducibility.py`:

```python
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
```

- [ ] **Step 4: Run the test, confirm pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_reproducibility.py
```
Expected: `All reproducibility tests passed.`, exit 0.

- [ ] **Step 5: Commit**

```powershell
git add utils/reproducibility.py test_reproducibility.py
git commit -m "feat: add reproducibility.set_seeds with test"
```

---

### Task 2: `utils/metrics.py` + test

**Files:**
- Create: `utils/metrics.py`
- Create: `test_metrics.py`

**Interfaces:**
- Consumes: nothing (sklearn only).
- Produces: `compute_metrics(y_true, y_score) -> dict` with keys `accuracy, precision, recall, f1, average_precision, auroc, confusion_matrix({tn,fp,fn,tp})`. `auroc`/`average_precision` are `None` for single-class input. Raises `ValueError` on non-binary labels or empty input.

- [ ] **Step 1: Write the test**

Create `test_metrics.py`:

```python
"""Standalone test for utils.metrics (no pytest; run with python)."""
import math

from utils.metrics import compute_metrics


def _assert_no_nan(m):
    for k, v in m.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"{k} is NaN"


def test_perfect_separation():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m['auroc'] == 1.0
    assert m['accuracy'] == 1.0
    assert m['confusion_matrix'] == {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}
    _assert_no_nan(m)


def test_confusion_matrix_mixed():
    # preds: 0.1->0, 0.9->1, 0.2->0, 0.8->1  vs true 0,0,1,1
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert m['confusion_matrix'] == {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1}
    assert abs(m['accuracy'] - 0.5) < 1e-9


def test_single_class_returns_none_not_nan():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert m['auroc'] is None
    assert m['average_precision'] is None
    assert m['accuracy'] == 1.0  # all predicted positive, all truly positive
    _assert_no_nan(m)


def test_non_binary_raises():
    try:
        compute_metrics([0, 1, 2], [0.1, 0.5, 0.9])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-binary labels")


def test_empty_raises():
    try:
        compute_metrics([], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty input")


if __name__ == '__main__':
    test_perfect_separation()
    test_confusion_matrix_mixed()
    test_single_class_returns_none_not_nan()
    test_non_binary_raises()
    test_empty_raises()
    print("All metrics tests passed.")
```

- [ ] **Step 2: Run it, confirm it fails on import**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_metrics.py
```
Expected: `ModuleNotFoundError: No module named 'utils.metrics'`.

- [ ] **Step 3: Implement `utils/metrics.py`**

Create `utils/metrics.py`:

```python
"""Binary-classification metrics, computed once per evaluation pass.

Replaces ad-hoc per-batch AUROC. Accumulate y_true / y_score across all
batches, then call compute_metrics ONCE — a single-class *batch* can no
longer corrupt the score. Unlike sklearn's per-call behaviour on
single-class input (which returns NaN), ranking metrics degrade to None.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_score):
    """Binary metrics from true labels and positive-class probabilities.

    Args:
        y_true: 1-D array-like of 0/1 integer labels.
        y_score: 1-D array-like of positive-class (class 1) probabilities in
            [0, 1] — e.g. F.softmax(outputs, dim=1)[:, 1].

    Returns:
        dict: accuracy, precision, recall, f1, average_precision, auroc,
        confusion_matrix ({tn, fp, fn, tp}). auroc and average_precision are
        None when y_true has a single class. Never returns NaN.

    Raises:
        ValueError: if y_true is empty or contains labels outside {0, 1}.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0:
        raise ValueError("compute_metrics: y_true is empty")
    labels_present = set(np.unique(y_true).tolist())
    if not labels_present.issubset({0, 1}):
        raise ValueError(
            f"compute_metrics expects binary labels in {{0,1}}, got {sorted(labels_present)}"
        )

    y_pred = (y_score >= 0.5).astype(int)
    both = labels_present == {0, 1}
    auroc = float(roc_auc_score(y_true, y_score)) if both else None
    ap = float(average_precision_score(y_true, y_score)) if both else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'average_precision': ap,
        'auroc': auroc,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }
```

- [ ] **Step 4: Run the test, confirm pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_metrics.py
```
Expected: `All metrics tests passed.`, exit 0.

- [ ] **Step 5: Commit**

```powershell
git add utils/metrics.py test_metrics.py
git commit -m "feat: add single-class-safe compute_metrics with test"
```

---

### Task 3: `utils/run_logger.py` + test + gitignore `runs/`

**Files:**
- Create: `utils/run_logger.py`
- Create: `test_run_logger.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `start_run(opt, root='runs') -> run_dir(str)`; `log_epoch(run_dir, epoch, metrics) -> None`; `finalize(run_dir, final_metrics) -> None`. Writes `config.json`, `metrics.json` (`{final, per_epoch}`), `metrics.csv` under `run_dir`. The `confusion_matrix` nested dict is flattened to `confusion_matrix_tn` etc. in json rows/csv.

- [ ] **Step 1: Write the test**

Create `test_run_logger.py`:

```python
"""Standalone test for utils.run_logger (no pytest; run with python)."""
import json
import os
import shutil
import tempfile
from argparse import Namespace

from utils import run_logger


def test_start_run_writes_config():
    root = tempfile.mkdtemp()
    try:
        opt = Namespace(seed=7, arch='efficientnet-b0', resume='')
        run_dir = run_logger.start_run(opt, root=root)
        assert os.path.isdir(run_dir)
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        assert cfg['seed'] == 7 and cfg['arch'] == 'efficientnet-b0'
    finally:
        shutil.rmtree(root)


def test_epoch_logging_and_finalize():
    root = tempfile.mkdtemp()
    try:
        run_dir = run_logger.start_run(Namespace(seed=1), root=root)
        m1 = {'auroc': 0.8, 'confusion_matrix': {'tn': 1, 'fp': 0, 'fn': 0, 'tp': 1}}
        m2 = {'auroc': 0.9, 'confusion_matrix': {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}}
        run_logger.log_epoch(run_dir, 1, m1)
        run_logger.log_epoch(run_dir, 2, m2)
        run_logger.finalize(run_dir, m2)
        with open(os.path.join(run_dir, 'metrics.json')) as f:
            data = json.load(f)
        assert len(data['per_epoch']) == 2
        assert data['final']['auroc'] == 0.9
        assert data['final']['confusion_matrix_tp'] == 2  # flattened
        assert os.path.exists(os.path.join(run_dir, 'metrics.csv'))
    finally:
        shutil.rmtree(root)


def test_finalize_without_epochs():
    root = tempfile.mkdtemp()
    try:
        run_dir = run_logger.start_run(Namespace(seed=1), root=root)
        run_logger.finalize(run_dir, {'auroc': 0.7, 'accuracy': 0.6})
        with open(os.path.join(run_dir, 'metrics.json')) as f:
            data = json.load(f)
        assert data['per_epoch'] == []
        assert data['final']['auroc'] == 0.7
        assert os.path.exists(os.path.join(run_dir, 'metrics.csv'))
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_start_run_writes_config()
    test_epoch_logging_and_finalize()
    test_finalize_without_epochs()
    print("All run_logger tests passed.")
```

- [ ] **Step 2: Run it, confirm it fails on import**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_run_logger.py
```
Expected: `ModuleNotFoundError: No module named 'utils.run_logger'`.

- [ ] **Step 3: Implement `utils/run_logger.py`**

Create `utils/run_logger.py`:

```python
"""Per-run artifacts: config + metrics to runs/<timestamp>/."""
import csv
import json
import os
from datetime import datetime


def start_run(opt, root='runs'):
    """Create runs/<timestamp>/ and dump the full options namespace.

    Returns the run directory path. Call once at the start of a run.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({k: _jsonable(v) for k, v in vars(opt).items()}, f, indent=2)
    return run_dir


def log_epoch(run_dir, epoch, metrics):
    """Append one epoch's (flattened) metrics to the per-epoch record."""
    path = os.path.join(run_dir, '_epochs.json')
    epochs = []
    if os.path.exists(path):
        with open(path) as f:
            epochs = json.load(f)
    epochs.append({'epoch': epoch, **_flatten(metrics)})
    with open(path, 'w') as f:
        json.dump(epochs, f, indent=2)


def finalize(run_dir, final_metrics):
    """Write metrics.json ({final, per_epoch}) and metrics.csv."""
    epochs_path = os.path.join(run_dir, '_epochs.json')
    per_epoch = []
    if os.path.exists(epochs_path):
        with open(epochs_path) as f:
            per_epoch = json.load(f)
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump({'final': _flatten(final_metrics), 'per_epoch': per_epoch}, f, indent=2)
    rows = per_epoch if per_epoch else [{'epoch': 0, **_flatten(final_metrics)}]
    fieldnames = sorted({k for r in rows for k in r})
    with open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _flatten(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub, subv in v.items():
                out[f'{k}_{sub}'] = subv
        else:
            out[k] = v
    return out


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
```

- [ ] **Step 4: Run the test, confirm pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_run_logger.py
```
Expected: `All run_logger tests passed.`, exit 0.

- [ ] **Step 5: Add `runs/` to `.gitignore`**

Append a line `runs/` to `.gitignore` (only if not already present).

- [ ] **Step 6: Commit**

```powershell
git add utils/run_logger.py test_run_logger.py .gitignore
git commit -m "feat: add run_logger for per-run config/metrics export"
```

---

### Task 4: Add `--seed` to options

`--manual_seed` exists in both `options/base.py:28` and `options/transfer.py:31` but is never consumed (verified: only declarations exist). Add `--seed` (default 7) to both; leave `manual_seed` untouched (harmless).

**Files:**
- Modify: `options/base.py`
- Modify: `options/transfer.py`

- [ ] **Step 1: Add `--seed` to `options/base.py`**

In `options/base.py`, immediately after the line `parser.add_argument('--manual_seed', type=int, default=7)`, add:
```python
        parser.add_argument('--seed', type=int, default=7,
                           help='Global RNG seed (passed to set_seeds)')
```

- [ ] **Step 2: Add `--seed` to `options/transfer.py`**

In `options/transfer.py`, immediately after the line `parser.add_argument('--manual_seed', type=int, default=7)`, add the identical line:
```python
        parser.add_argument('--seed', type=int, default=7,
                           help='Global RNG seed (passed to set_seeds)')
```

- [ ] **Step 3: Verify both parse**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import sys; sys.argv=['t','--seed','42']; from options.base import BaseOptions; print('base', BaseOptions().parse(print_options=False).seed)"
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import sys; sys.argv=['t','--target','style2','--seed','42']; from options.transfer import BaseOptions; print('transfer', BaseOptions().parse(print_options=False).seed)"
```
Expected: `base 42` and `transfer 42`.

- [ ] **Step 4: Commit**

```powershell
git add options/base.py options/transfer.py
git commit -m "feat: add --seed option to base and transfer parsers"
```

---

### Task 5: Wire eval.py + quick_start.py (defines the canonical pattern)

Both are eval-only scripts using `TestOptions`. Replace the per-batch `arc` AUROC meter with end-of-epoch accumulation through `compute_metrics`, seed at startup, and export one run.

**CANONICAL WIRING PATTERN (referenced by Tasks 6–8):**

*Imports (top of file, after existing imports):*
```python
import torch.nn.functional as F
from utils.reproducibility import set_seeds
from utils.metrics import compute_metrics
from utils import run_logger
```
*After `opt = ...parse(...)` and before model/data construction:*
```python
set_seeds(opt.seed)
```
*Inside an eval/validation function: accumulate instead of per-batch AUROC.* Replace the `arc = AverageMeter()` usage and the per-batch `auroc = roc_auc_score(...)` / `arc.update(...)` lines with:
```python
    y_true_all, y_score_all = [], []
    # ... inside the batch loop, after `outputs = model(inputs)`:
            probs = F.softmax(outputs, dim=1)[:, 1]
            y_true_all.append(targets.detach().cpu())
            y_score_all.append(probs.detach().cpu())
    # ... after the loop:
    metrics = compute_metrics(torch.cat(y_true_all).numpy(),
                              torch.cat(y_score_all).numpy())
```
*Printing:* print the metric set, e.g.:
```python
    print("acc:{accuracy:.4f} prec:{precision:.4f} rec:{recall:.4f} "
          "f1:{f1:.4f} ap:{average_precision} auroc:{auroc}".format(**metrics))
```
(`auroc`/`average_precision` may be `None`; `{auroc}` prints `None` cleanly — do NOT apply `:.4f` to them.)

**Files:**
- Modify: `eval.py`
- Modify: `quick_start.py`

- [ ] **Step 1: Wire `eval.py`**

Apply the canonical pattern to `eval.py`'s `test()` function (the `arc`/`roc_auc_score`/`arc.update` block at lines ~71-82). `test()` should build and return the `metrics` dict (keep returning `(losses.avg, metrics)` or just `metrics` — update both call sites at lines ~98 and ~106 accordingly). After parsing, add `set_seeds(opt.seed)` and `run_dir = run_logger.start_run(opt)`. Eval runs both source and target; call `run_logger.finalize(run_dir, metrics)` once after the final (target) eval with that eval's metrics. `import torch` is already present.

- [ ] **Step 2: Smoke-verify `eval.py`**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe eval.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar --resume weights/t-gd/efficientnet/star_to_style2.pth.tar --test_batch 16 --num_workers 0 --size 64
```
Expected: weights load, two metric lines printed with `acc/prec/rec/f1/ap/auroc` values, NO `nan`, exit 0. Then check a run dir was written:
```powershell
Get-ChildItem runs | Select-Object -Last 1 | ForEach-Object { Get-ChildItem $_.FullName }
```
Expected: `config.json`, `metrics.json`, `metrics.csv`.

- [ ] **Step 3: Wire `quick_start.py`**

Apply the same canonical pattern to `quick_start.py`'s `test()` (same `arc`/`roc_auc_score` structure at lines ~71-82). Add `set_seeds(opt.seed)` + `run_logger.start_run` after parse; `finalize` after the final eval. `import torch` is present.

- [ ] **Step 4: Smoke-verify `quick_start.py`**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe quick_start.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 --resume weights/pre-train/efficientnet/stargan.pth.tar
```
Expected: weights load, two metric lines, NO `nan`, exit 0, a new `runs/<ts>/` with the three files.

- [ ] **Step 5: Commit**

```powershell
git add eval.py quick_start.py
git commit -m "feat: wire seeds, compute_metrics, and run export into eval and quick_start"
```

---

### Task 6: Wire pretrain.py + early_stop_pretrain.py

Both train and validate, using `options/base.py`. Apply the canonical pattern from Task 5 to their validation/`test()` paths. Keep training-loss/top-1 running meters; only the AUROC computation moves to end-of-epoch via `compute_metrics`. Per epoch call `run_logger.log_epoch(run_dir, epoch, val_metrics)`; after the loop call `run_logger.finalize(run_dir, last_val_metrics)`.

**Files:**
- Modify: `pretrain.py`
- Modify: `early_stop_pretrain.py`

- [ ] **Step 1: Read both files' validation paths**

Identify each script's per-batch AUROC block (search for `roc_auc_score` and `arc`). Confirm where the epoch loop and checkpoint save are (for placing `set_seeds`, `start_run`, `log_epoch`, `finalize`).

- [ ] **Step 2: Wire `pretrain.py`**

Apply the canonical pattern. `set_seeds(opt.seed)` + `run_dir = run_logger.start_run(opt)` after `opt = BaseOptions().parse(...)`. Validation function accumulates y_true/y_score → `compute_metrics`; print includes the metric set. In the epoch loop, `run_logger.log_epoch(run_dir, epoch, val_metrics)`; after the loop, `run_logger.finalize(run_dir, val_metrics)`.

- [ ] **Step 3: Smoke-verify `pretrain.py`**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe pretrain.py --source_dataset dataset/_smoke --epochs 2 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_pretrain
```
Expected: 2 epochs of train/val output, per-epoch metric set, NO `nan`, exit 0; `log/smoke_pretrain/checkpoint.pth.tar` exists; a `runs/<ts>/` with config/metrics.json/metrics.csv where `per_epoch` has 2 rows.

- [ ] **Step 4: Wire `early_stop_pretrain.py`**

Apply the same pattern. Place `finalize` after the training loop ends (including the early-stop break path — ensure `finalize` runs regardless of how the loop exits, e.g. after the loop, not inside it).

- [ ] **Step 5: Smoke-verify `early_stop_pretrain.py`**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe early_stop_pretrain.py --source_dataset dataset/_smoke --epochs 2 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_es
```
Expected: 2 epochs (or early stop), per-epoch metric set, NO `nan`, exit 0; `log/smoke_es/checkpoint.pth.tar` exists; `runs/<ts>/` written.

- [ ] **Step 6: Commit**

```powershell
git add pretrain.py early_stop_pretrain.py
git commit -m "feat: wire seeds, compute_metrics, and run export into pretrain scripts"
```

---

### Task 7: Wire synermix_pretrain.py

Uses `options/base.py` and the `main()` from Phase 1. Apply the canonical pattern to its `test()` validation path. `set_seeds(opt.seed)` + `start_run` at the top of `main()`; `log_epoch` per epoch in the loop; `finalize` after the loop.

**Files:**
- Modify: `synermix_pretrain.py`

- [ ] **Step 1: Read `main()` and `test()`**

Locate the per-batch AUROC in `test()` (search `roc_auc_score`/`arc`) and the epoch loop + checkpoint save in `main()`.

- [ ] **Step 2: Wire it**

Apply the canonical pattern. Imports at top of file. In `main()`: `set_seeds(opt.seed)` and `run_dir = run_logger.start_run(opt)` near the top (after `model`/`device` setup is fine, but before training). `test()` accumulates → `compute_metrics`, returns/prints the metric set. In the loop: `run_logger.log_epoch(run_dir, epoch, test_metrics)`. After the loop: `run_logger.finalize(run_dir, test_metrics)`.

- [ ] **Step 3: Smoke-verify `synermix_pretrain.py`**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe synermix_pretrain.py --source_dataset dataset/_smoke --epochs 2 --synermix_warmup_epochs 1 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke
```
Expected: `Epoch: [1 | 2]` then `[2 | 2]` (SynerMix path), per-epoch metric set, NO `nan`, exit 0; `log/smoke/checkpoint.pth.tar` + `model_best.pth.tar` exist; `runs/<ts>/` written with 2 per-epoch rows.

- [ ] **Step 4: Verify import-safety preserved**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: `All tests passed successfully!` (importing the module must NOT start training or a run).

- [ ] **Step 5: Commit**

```powershell
git add synermix_pretrain.py
git commit -m "feat: wire seeds, compute_metrics, and run export into synermix_pretrain"
```

---

### Task 8: Wire transfer.py

Uses `options/transfer.py`. Phase 1 left a guarded per-batch AUROC in both `train()` and `test()`; replace the `test()` one with accumulation→`compute_metrics` (the guard becomes unnecessary there). Leave `train()`'s in place OR also convert it — converting is preferred for consistency, but `test()` is the validation metric that matters; if converting `train()` is risky, leave its Phase-1 guard. `set_seeds(opt.seed)` + `start_run` after parse; `log_epoch` per epoch; `finalize` after the loop.

**Files:**
- Modify: `transfer.py`

- [ ] **Step 1: Wire `test()` + startup + loop**

Imports at top. After `opt = BaseOptions().parse(...)`: `set_seeds(opt.seed)` and `run_dir = run_logger.start_run(opt)`. Convert `test()` to accumulate y_true/y_score → `compute_metrics`; print the metric set. transfer runs `test()` twice per epoch (target val + source val) — capture both metric dicts; `run_logger.log_epoch(run_dir, epoch, {'target': target_metrics, 'val_source': source_metrics})` will flatten via the nested-dict flattener into `target_*`/`val_source_*` columns. After the loop, `run_logger.finalize(run_dir, target_metrics)`.

- [ ] **Step 2: Smoke-verify `transfer.py` (run twice — unseeded-shuffle edge no longer applies, but confirm stability)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe transfer.py --target style2 --source_dataset dataset/_smoke --target_dataset dataset/_smoke --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar --epochs 1 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_transfer
```
Expected: weights load, one epoch, metric set for target + source val, NO `nan`, exit 0; `log/smoke_transfer/checkpoint.pth.tar` exists; `runs/<ts>/` written.

- [ ] **Step 3: Commit**

```powershell
git add transfer.py
git commit -m "feat: wire seeds, compute_metrics, and run export into transfer"
```

---

### Task 9: Pin requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Capture installed versions**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pip freeze
```
Note the exact versions for: torch, torchvision, scikit-learn, scipy, numpy, opencv-python, Pillow, matplotlib, tqdm, torchsummary, progress.

- [ ] **Step 2: Rewrite `requirements.txt` pinned + curated**

Replace the file contents with pinned, curated direct deps. For `torch`/`torchvision`, strip the local `+cpu` build tag and pin the base version with a comment pointing at the CPU wheel index (so a fresh install resolves). Template (fill exact versions from Step 1):

```
# Pinned to the working CPU venv (Python 3.12). For CPU torch wheels:
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
torch==2.12.0
torchvision==0.27.0
scikit-learn==<from pip freeze>
scipy==<from pip freeze>
numpy==<from pip freeze>
opencv-python==<from pip freeze>
Pillow==<from pip freeze>
matplotlib==<from pip freeze>
tqdm==<from pip freeze>
torchsummary==<from pip freeze>
progress==<from pip freeze>
git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

- [ ] **Step 3: Verify pins match installed**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pip install --dry-run -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
Expected: pip reports requirements already satisfied / resolvable, no version conflicts. (If `--dry-run` is unavailable in this pip, instead spot-check each pin with `pip show <pkg>` and confirm versions match.)

- [ ] **Step 4: Commit**

```powershell
git add requirements.txt
git commit -m "build: pin requirements.txt to working CPU venv versions"
```

---

### Task 10: README rewrite

Replace the upstream T-GD README with a SynerMix-focused one. Author = Vincent Ferrer. Use `<!-- FILL: ... -->` for the thesis title and result cells; derive the method summary from `synermix_pretrain.py`. Preserve the T-GD citation.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Capture the T-GD citation from git history**

The current `README.md` top lines contain the T-GD ICML 2020 citation — preserve that text for the Credits section before overwriting.

- [ ] **Step 2: Write the new `README.md`**

Use this skeleton (fill derived content; leave `<!-- FILL -->` for thesis specifics):

```markdown
# SynerMix <!-- FILL: full thesis title -->

**Author:** Vincent Ferrer

<!-- FILL after Phase 3/4: live-demo badge, CI badge -->

Detecting GAN-generated face images with a transferable detector, extended with
**SynerMix** — a synergistic mixing augmentation. A fork of T-GD (ICML 2020).

## What is this

GAN-generated images leave subtle fingerprints. T-GD trains an EfficientNet-b0
binary detector (real vs. generated) that transfers across GAN families via
L2-SP self-training. This repo adds SynerMix, applied during pretraining.

## SynerMix — the contribution

SynerMix augments pretraining by combining two mixing strategies under a
dynamic schedule (see `synermix_pretrain.py`):

- **Intra-class feature mixing** — weighted feature-space blends within a class
  enrich representation without crossing the decision boundary.
- **Inter-class CutMix** — area-corrected, per-sample λ CutMix across classes
  for boundary regularization.
- **Dynamic β schedule** — a warm-up (`--synermix_warmup_epochs`) defers mixing
  until the backbone stabilizes, then `--synermix_beta` balances the two.

## Results

Numbers as reported in the thesis (regenerate with `eval.py` after downloading
a real test set — see Quickstart).

| Transfer | AUROC | Accuracy |
|----------|-------|----------|
| StarGAN → StyleGAN2 | <!-- FILL --> | <!-- FILL --> |
| <!-- FILL: add rows --> | | |

## Quickstart (Windows / PowerShell)

```powershell
# 1. Environment
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Weights (already in weights/) — or run weights/download_weights.sh

# 3. Zero-download sanity check: synthetic smoke dataset
$env:PYTHONUTF8='1'; python scripts\make_smoke_dataset.py

# 4. Evaluate bundled weights
$env:PYTHONUTF8='1'; python quick_start.py --source_dataset dataset/_smoke `
  --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 `
  --resume weights/pre-train/efficientnet/stargan.pth.tar
```

> Windows notes: always set `$env:PYTHONUTF8='1'` (scripts print Unicode) and
> pass `--num_workers 0` (train transforms aren't picklable by Windows workers).

## Repo layout

| Path | Role |
|------|------|
| `synermix_pretrain.py` | SynerMix pretraining (headline contribution) |
| `pretrain.py`, `early_stop_pretrain.py` | Baseline pretraining |
| `transfer.py` | L2-SP transfer / self-training |
| `eval.py`, `quick_start.py` | Evaluation |
| `models/` | Canonical EfficientNet (GroupNorm) |
| `resnext/` | ResNeXt backbone (ensemble mode in the UI) |
| `utils/` | metrics, reproducibility, run logging, helpers |
| `scripts/` | smoke-dataset generator |
| `notebooks/` | archived training notebook |

## Credits

Built on **T-GD: Transferable GAN-generated Images Detection Framework** —
Hyeonseong Jeon, Youngoh Bang, Junyaup Kim, Simon S. Woo. ICML 2020.
Upstream: https://github.com/cutz-j/T-GD
```

(Adjust the result-table rows to the transfer weights actually present in `weights/t-gd/efficientnet/`.)

- [ ] **Step 3: Sanity-check rendering + placeholders**

Confirm no fabricated numbers remain (all thesis specifics are `<!-- FILL -->`), and quickstart flags match the repo.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: rewrite README around SynerMix contribution"
```

---

### Task 11: Final verification — determinism + full smoke matrix

**Files:** none (verification only).

- [ ] **Step 1: All unit tests pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_reproducibility.py
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_metrics.py
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_run_logger.py
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: each prints its pass line, exit 0.

- [ ] **Step 2: Seeded determinism — two eval runs match**

Run `quick_start.py` twice with the same `--seed 7` on `dataset/_smoke`, then compare the `final` metrics of the two newest `runs/` dirs:
```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe quick_start.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 --seed 7 --resume weights/pre-train/efficientnet/stargan.pth.tar
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe quick_start.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 --seed 7 --resume weights/pre-train/efficientnet/stargan.pth.tar
$two = Get-ChildItem runs | Sort-Object Name | Select-Object -Last 2
$a = Get-Content (Join-Path $two[0].FullName 'metrics.json') | ConvertFrom-Json
$b = Get-Content (Join-Path $two[1].FullName 'metrics.json') | ConvertFrom-Json
"auroc A=$($a.final.auroc) B=$($b.final.auroc)  acc A=$($a.final.accuracy) B=$($b.final.accuracy)"
```
Expected: the two runs report identical `auroc` and `accuracy`.

- [ ] **Step 3: Full entry-point smoke matrix (no `nan`, exit 0 each)**

Rerun all six entry points using the commands in Tasks 5–8 (eval, quick_start, pretrain, early_stop_pretrain, synermix_pretrain, transfer). Confirm each exits 0, prints the metric set, writes its checkpoint where applicable, and shows no `nan`.

- [ ] **Step 4: Clean tree**

```powershell
git status --short
```
Expected: only `?? PORTFOLIO_ITEMS.md` untracked (`runs/`, `log/`, `dataset/` gitignored). Report any deviation.
