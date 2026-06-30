# Phase 4: Trust Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real pytest suite, clean ruff lint across the whole repo (fixing two real latent bugs along the way), GitHub Actions CI with a README badge, and a CLAUDE.md.

**Architecture:** Add `pytest` + `ruff` as pinned dev deps and a `pyproject.toml` holding both tools' config. Existing `test_*.py` are already pytest-discoverable; weight/data-dependent tests get skip guards so CI (which has no `weights/`/`dataset/`) stays green. Ruff runs repo-wide (notebooks excluded): auto-fix the safe ~99, then manually resolve the rest. CI installs CPU torch, runs `ruff check` + `pytest`.

**Tech Stack:** Python 3.12, pytest, ruff 0.15.x, PyTorch 2.12.0+cpu. GitHub Actions (ubuntu-latest). Windows/PowerShell locally.

## Global Constraints

- Local Python runs: prefix `$env:PYTHONUTF8='1'; `, use `.\venv\Scripts\python.exe`, run from repo root.
- pytest tests must pass BOTH with `weights/`/`dataset/` present (local) AND absent (CI) — absent → skip, never fail.
- Ruff: default ruleset (do NOT add E501/line-length), `exclude = ["notebooks", "venv"]`, `target-version = "py312"`. Final gate: `ruff check .` exits 0.
- `weights/` and `dataset/` are gitignored — never commit them; regenerate the smoke set with `scripts/make_smoke_dataset.py`.
- Review the `ruff --fix` diff before committing — autofix can remove an import that is intentionally re-exported via a star-import facade.

---

### Task 1: pytest setup + skip guards

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`, `test_detector.py`

- [ ] **Step 1: Install pytest and capture the version**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pip install pytest --quiet; .\venv\Scripts\python.exe -m pip show pytest | Select-String '^Version'
```
Note the version (e.g. `Version: 8.x.y`).

- [ ] **Step 2: Create `pyproject.toml` with the pytest config**

Create `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["."]
python_files = "test_*.py"
addopts = "-q"
```

- [ ] **Step 3: Add skip guard to `test_detector.py`**

In `test_detector.py`, after the `CKPT = ...` / `CPU = ...` lines, add:
```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT), reason="bundled weights not present (e.g. CI)"
)
```
(Keep the existing `if __name__ == '__main__':` runner — when run standalone with weights present it still works; under pytest the marker skips the whole module if the checkpoint is absent.)

- [ ] **Step 4: Verify pytest discovers and runs (weights present locally)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q
```
Expected: all test files collected; `test_detector`/`test_metrics`/`test_reproducibility`/`test_run_logger`/`test_synermix` pass (the detector tests RUN because weights are present). 0 failures. (May take ~2-3 min — `test_synermix` and `test_detector` build EfficientNet.)

- [ ] **Step 5: Verify the CI case — tests SKIP (not fail) when weights are absent**

```powershell
Rename-Item weights weights_hidden
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q test_detector.py
Rename-Item weights_hidden weights
```
Expected: `test_detector.py` reports `skipped` (not failed), exit 0. (The rename simulates CI's missing weights. Restoring the dir is the last step — make sure it runs.)

- [ ] **Step 6: Pin pytest in `requirements.txt`**

Add to `requirements.txt` (use the exact version from Step 1), in a small dev-tools group at the end:
```
# dev / CI
pytest==<from Step 1>
```

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml requirements.txt test_detector.py
git commit -m "test: add pytest config and CI-safe skip guards"
```

---

### Task 2: Ruff config + auto-fixes

**Files:**
- Modify: `pyproject.toml`, `requirements.txt`

- [ ] **Step 1: Confirm ruff is installed and capture the version**

```powershell
.\venv\Scripts\ruff.exe --version
```
Note it (e.g. `ruff 0.15.20`). If missing: `.\venv\Scripts\python.exe -m pip install ruff`.

- [ ] **Step 2: Add the ruff config to `pyproject.toml`**

Append to `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py312"
exclude = ["notebooks", "venv"]

[tool.ruff.lint]
# Ruff defaults (pyflakes F + pycodestyle E/W subset). Line-length (E501) is
# intentionally NOT enabled to avoid reflowing the forked legacy files.
```

- [ ] **Step 3: Baseline the finding count**

```powershell
.\venv\Scripts\ruff.exe check . 2>&1 | Select-String 'Found '
```
Expected: ~`Found 157 errors.` (notebooks now excluded may lower it; record the number).

- [ ] **Step 4: Apply safe auto-fixes**

```powershell
.\venv\Scripts\ruff.exe check . --fix
```
Then **review the diff** (`git diff`) — confirm no intentionally re-exported import was removed from `utils/*.py` (the star-import facade). If autofix removed a name that `utils/__init__.py` re-exports, restore it (it will be made explicit in Task 3). Re-baseline:
```powershell
.\venv\Scripts\ruff.exe check . 2>&1 | Select-String 'Found '
```
Expected: the count drops substantially (~58 remaining, the non-auto-fixable ones).

- [ ] **Step 5: Pin ruff in `requirements.txt`**

Add under the dev group:
```
ruff==<from Step 1>
```

- [ ] **Step 6: Verify nothing broke (import-safe modules still import)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import utils, detector; from utils import Bar, Logger, AverageMeter, accuracy; print('utils facade + detector import OK')"
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m py_compile app.py gradio_ui.py launch_ui.py eval.py quick_start.py pretrain.py transfer.py synermix_pretrain.py early_stop_pretrain.py; echo "compile EXIT=$LASTEXITCODE"
```
Expected: `utils facade + detector import OK` and `compile EXIT=0`. (If a removed import broke a re-export, fix before committing.)

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml requirements.txt
git add -u
git commit -m "style: add ruff config and apply safe auto-fixes"
```

---

### Task 3: Ruff manual fixes — real bugs + remainder

Resolve the ~58 non-auto-fixable findings so `ruff check .` exits 0. This includes two real latent bugs. Work category by category; re-run `ruff check .` after each to track progress.

**Files:**
- Modify: `utils/misc.py`, `resnext/model.py`, `utils/__init__.py`, `utils/visualize.py`, `utils/logger.py`, `utils/eval.py`, and whichever files still carry F841/E-series findings (e.g. `eval.py`, `quick_start.py`, `synermix_pretrain.py`, `early_stop_pretrain.py`, `gradio_ui.py`).

- [ ] **Step 1: Fix real bug #1 — missing torch import in `utils/misc.py`**

`get_mean_and_std()` uses `torch` but the module never imports it (F821). At the top of `utils/misc.py`, ensure `import torch` is present (add it if absent). Also fix the `dataloader = trainloader = torch...` double-assignment F841: change to `dataloader = torch.utils.data.DataLoader(...)` and drop the unused `trainloader` name.

- [ ] **Step 2: Fix real bug #2 — undefined `model_urls` in `resnext/model.py`**

At `resnext/model.py:~229-233`, the `if pretrained:` branch references undefined `model_urls[arch]`. The repo never builds resnext from a URL (always `num_classes=2`), so this branch is dead. Remove the `if pretrained:` download branch (and the now-unused `pretrained`/`progress` handling if they become dead), OR if `pretrained` is part of a shared signature, raise `NotImplementedError("pretrained download not supported")` instead of referencing the undefined name. Prefer removing the dead branch (YAGNI). Re-run `ruff check resnext/model.py` → no F821.

- [ ] **Step 3: Resolve the star-import facade (F403/F405/remaining F821)**

For each of `utils/misc.py`, `utils/logger.py`, `utils/visualize.py`, `utils/eval.py`, add an explicit `__all__` listing the public names that `utils/__init__.py` re-exports (the names other modules import via `from utils import ...`, e.g. `Bar`, `Logger`, `AverageMeter`, `accuracy`, `mkdir_p`, `savefig`, `progress_bar`, etc. — read each module to enumerate its public functions/classes). Keep the `from .x import *` lines in `utils/__init__.py` and `utils/visualize.py`. If a module's public surface is large/unclear, instead annotate its star-import line with `# noqa: F403` and the using line with `# noqa: F405`, with a one-line comment explaining the intentional facade. Goal: `ruff check utils/` clean.

- [ ] **Step 4: Resolve remaining F841 unused vars and E-series**

- F841 (e.g. `prec1` in `eval.py`/`quick_start.py`, `intra_prec1`/`batch_size` in `synermix_pretrain.py`, `improvement` in `early_stop_pretrain.py`, `current_step`/accordion vars in `gradio_ui.py`): delete the assignment, or use the value. Do NOT change behavior — these are computed-then-discarded; deleting the assignment is safe.
- E711 (`== None`): change to `is None`.
- E701 (compound one-liners): split onto separate lines.
- E721 (`type(x) == Y`): use `isinstance(x, Y)` or `type(x) is Y` as appropriate.
- E402 (import not at top): move to top, or add `# noqa: E402` where a late import is intentional (e.g. right after a `sys.path.append`).
- E401 / F811 if any remain: split imports / remove the redefinition.

- [ ] **Step 5: Verify ruff is clean**

```powershell
.\venv\Scripts\ruff.exe check . ; echo "ruff EXIT=$LASTEXITCODE"
```
Expected: `All checks passed!` and `ruff EXIT=0`.

- [ ] **Step 6: Verify no behavior regression**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import utils, detector; from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig; print('utils facade OK')"
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe eval.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar --resume weights/t-gd/efficientnet/star_to_style2.pth.tar --test_batch 16 --num_workers 0 --size 64 2>&1 | Select-String 'acc:|nan|Error'
```
Expected: facade imports OK; pytest passes (no failures); `eval.py` prints metric lines with NO `nan`, exit 0. (If `weights/` was renamed in a prior task, ensure it's restored first.)

- [ ] **Step 7: Commit**

```powershell
git add -u
git commit -m "fix: resolve remaining ruff findings incl. missing torch import and dead resnext url branch"
```

---

### Task 4: GitHub Actions CI + README badge

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install dependencies (CPU torch)
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
      - name: Lint (ruff)
        run: ruff check .
      - name: Test (pytest)
        run: pytest -q
```

- [ ] **Step 2: Validate the YAML**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8')); print('ci.yml valid YAML')"
```
Expected: `ci.yml valid YAML`. (If `yaml` isn't importable, `pip install pyyaml` first — it's a transitive dep of many packages and usually present.)

- [ ] **Step 3: Confirm the CI commands match local reality**

Run exactly what CI runs (the gates), to be sure they pass:
```powershell
.\venv\Scripts\ruff.exe check . ; echo "ruff EXIT=$LASTEXITCODE"
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q
```
Expected: ruff EXIT=0; pytest passes (detector tests run locally since weights are present — in CI they skip).

- [ ] **Step 4: Add the badge to `README.md`**

Replace the Phase 2/3 badge placeholder comment near the top of `README.md` with (substitute the real GitHub owner/repo — `vinnyy-ph/Thesis-master`):
```markdown
[![CI](https://github.com/vinnyy-ph/Thesis-master/actions/workflows/ci.yml/badge.svg)](https://github.com/vinnyy-ph/Thesis-master/actions/workflows/ci.yml)
```

- [ ] **Step 5: Commit**

```powershell
git add .github/workflows/ci.yml README.md
git commit -m "ci: add GitHub Actions lint+test workflow and README badge"
```

---

### Task 5: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write `CLAUDE.md`**

Create `CLAUDE.md`:
```markdown
# CLAUDE.md

Guidance for working in this repo (T-GD GAN-image detector + the SynerMix thesis
contribution).

## Entry points

| Script | Role |
|--------|------|
| `synermix_pretrain.py` | SynerMix pretraining (the thesis contribution) |
| `pretrain.py`, `early_stop_pretrain.py` | Baseline pretraining |
| `transfer.py` | L2-SP transfer / self-training to a new GAN |
| `eval.py`, `quick_start.py` | Evaluation (metrics + run export) |
| `app.py` | Clean local Gradio demo (verdict + GradCAM) |
| `gradio_ui.py` | Full local UI (ensemble, multi-image) |
| `launch_ui.py` | Dependency-checking launcher for `gradio_ui.py` |

## Architecture

- `models/` is the **canonical** EfficientNet (GroupNorm, `_gn*` keys) — it matches
  the bundled checkpoints. `resnext/` is the ResNeXt backbone used by the UI's
  ensemble mode.
- `detector.py` is the single source of truth for loading a checkpoint
  (`load_detector`, with SynerMix state-dict remap + logged `weights_only`
  fallback), inference, and GradCAM.
- `utils/` holds shared helpers: `metrics.py` (single-class-safe `compute_metrics`),
  `reproducibility.py` (`set_seeds`), `run_logger.py` (per-run `runs/<ts>/` export),
  plus logging/eval/visualize utilities (re-exported via `utils/__init__.py`).

## Known pitfalls (Windows / CPU)

- Set `PYTHONUTF8=1` before running anything — scripts print Unicode that crashes
  the default cp1252 console.
- Pass `--num_workers 0` to training/eval scripts on Windows — the train transforms
  use `transforms.Lambda`, which DataLoader workers can't pickle.
- This is a CPU-only setup; checkpoint loads pass `map_location` (GPU-saved files
  otherwise fail).
- `weights/` and `dataset/` are gitignored. Get weights via
  `weights/download_weights.sh` (or the OneDrive link in the README); generate a
  tiny synthetic test set with `python scripts/make_smoke_dataset.py`.

## Testing

- `pytest` runs the suite (CI does this). Weight/data-dependent tests skip when
  those dirs are absent. Any `test_*.py` can also be run standalone:
  `python test_metrics.py`.
- Lint with `ruff check .`.
```

- [ ] **Step 2: Verify it renders / has no broken structure**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "t=open('CLAUDE.md',encoding='utf-8').read(); assert '## Entry points' in t and '## Known pitfalls' in t; print('CLAUDE.md OK', len(t), 'chars')"
```
Expected: `CLAUDE.md OK ... chars`.

- [ ] **Step 3: Commit**

```powershell
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with entry points, architecture, and pitfalls"
```

---

### Task 6: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Lint clean**

```powershell
.\venv\Scripts\ruff.exe check . ; echo "ruff EXIT=$LASTEXITCODE"
```
Expected: `All checks passed!`, EXIT=0.

- [ ] **Step 2: Full pytest (weights present → detector tests run)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q
```
Expected: all pass, 0 failures.

- [ ] **Step 3: CI-simulation (weights absent → skips, still green)**

```powershell
Rename-Item weights weights_hidden
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m pytest -q
Rename-Item weights_hidden weights
```
Expected: some skips (detector), 0 failures, exit 0. Confirm `weights` is restored afterward.

- [ ] **Step 4: Real-bug fixes confirmed**

```powershell
.\venv\Scripts\ruff.exe check utils/misc.py resnext/model.py --select F821 ; echo "F821 EXIT=$LASTEXITCODE"
```
Expected: `All checks passed!`, EXIT=0 (no undefined names in the two fixed files).

- [ ] **Step 5: Clean tree**

```powershell
git status --short
```
Expected: only `?? PORTFOLIO_ITEMS.md`. Report any deviation.
