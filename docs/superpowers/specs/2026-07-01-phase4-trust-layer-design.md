# Phase 4 — Trust Layer (pytest + CI + CLAUDE.md)

**Date:** 2026-07-01
**Status:** Approved
**Parent spec:** `2026-06-10-tgd-portfolio-sprint-design.md` (Phase 4)
**Prereq:** Phases 1–3 merged to master.

## Goal

Make the repo trustworthy at a glance: a real `pytest` suite, a green CI badge,
clean lint across the whole repo, and a `CLAUDE.md` that orients any
contributor (human or agent). This is the last main sprint phase.

## Decisions (from brainstorming, 2026-07-01)

- **Lint scope: whole repo, fix everything** (user choice). Ruff's default
  ruleset (pyflakes `F` + a pycodestyle `E`/`W` subset; **line-length E501 is NOT
  in defaults**, so no mass reformat). Measured baseline: **157 findings**
  (107 unused-import, 12 unused-var, 11 empty f-string, 6 undefined-name, 5
  star-import, plus a handful of E-series). `notebooks/` is **excluded** from
  lint (archived exploratory artifacts, not shipped code).
- **pytest** is the one new dependency. Weight/data-dependent tests skip
  gracefully when the (gitignored) `weights/` and `dataset/` are absent, so CI is
  green without them.
- **CI** runs on push + pull_request, Ubuntu, Python 3.12, CPU torch wheels.
- **CLAUDE.md** documents entry points, the canonical model package, and the
  known Windows/CPU pitfalls.

## Context (measured 2026-07-01)

- `weights/` and `dataset/` are gitignored — NOT in the repo. CI cannot load
  checkpoints or real data. The smoke fixture is regenerated on demand by
  `scripts/make_smoke_dataset.py`.
- Existing tests are 5 plain-`assert` scripts (`test_synermix.py`,
  `test_metrics.py`, `test_reproducibility.py`, `test_run_logger.py`,
  `test_detector.py`) — already function+assert style, pytest-discoverable.
- No `.github/workflows`, no pytest/ruff config, no `CLAUDE.md` yet.
- `ruff check .` (v0.15.x, default rules) → 157 errors, 99 auto-fixable.
- Two findings are **real latent bugs** (not style): `utils/misc.py`
  `get_mean_and_std()` uses `torch` without importing it (the function crashes if
  called); `resnext/model.py:231` references undefined `model_urls` in the
  dormant `pretrained=True` download branch.
- The `utils/` package is a **star-import facade** (`utils/__init__.py` does
  `from .misc import *` etc.; the whole codebase relies on `from utils import
  Bar, Logger, AverageMeter, ...`). This is the source of the 5 `F403` and
  contributes to false `F821`.

## Components

### 1. pytest suite

- Add `pytest==<pinned>` to `requirements.txt` (curated, pinned like the rest).
- Add `pyproject.toml` `[tool.pytest.ini_options]`: `testpaths = ["."]`,
  `python_files = "test_*.py"`, `addopts = "-q"`.
- Keep the existing test functions and their `if __name__ == '__main__':` runners
  (harmless; lets them still run standalone). No rewrite needed — they are already
  `test_*` functions with real asserts.
- Make weight/data-dependent tests CI-safe with module-level skip guards:
  - `test_detector.py`: `pytestmark = pytest.mark.skipif(not os.path.exists(CKPT),
    reason="bundled weights not present")` (the load/predict/gradcam tests all
    need the checkpoint).
  - Audit `test_synermix.py`: it builds an EfficientNet but needs no weights/data
    (operates on random tensors) — it runs in CI. If any part reads `dataset/` or
    a checkpoint, guard that part too. (Believed clean; verify during implementation.)
  - `test_metrics.py`, `test_reproducibility.py`, `test_run_logger.py`: no external
    deps — always run.
- Verification: `pytest` passes locally with weights present; `pytest` passes
  (with skips, no failures) when `weights/` is renamed away (simulating CI).

### 2. Ruff cleanup (whole repo, fix everything)

- Add `[tool.ruff]` to `pyproject.toml`: default select, `exclude = ["notebooks",
  "venv"]`, `target-version = "py312"`. Keep line-length default but **do not**
  add E501 to select (stay with ruff defaults) so the legacy files don't need
  reflowing.
- Run `ruff check . --fix` to auto-resolve the ~99 safe fixes (unused imports,
  empty f-strings, multiple-imports-on-one-line).
- Manually resolve the remainder:
  - **Real bugs:** add `import torch` to `utils/misc.py`; in `resnext/model.py`
    either define the standard `model_urls` dict or remove the dead
    `pretrained=True` branch (it is never used — the repo builds resnext with
    `num_classes=2`, not from a URL). Removing the dead branch is preferred (YAGNI).
  - **Star-import facade (`F403`/`F405`):** give `utils/misc.py`,
    `utils/logger.py`, `utils/visualize.py`, `utils/eval.py` explicit `__all__`
    lists and keep the `from .x import *` in `utils/__init__.py` (the re-export is
    intentional API). If an `__all__` proves impractical for a module, fall back to
    a commented `# noqa: F403` on that import line. Re-run ruff to confirm the
    `F403`/`F405`/`F821`-from-stars clear.
  - **Unused vars (`F841`):** delete or use each (e.g. `prec1` computed but unused,
    `trainloader` double-assignment, `current_step`); none are load-bearing.
  - **E-series:** `== None`→`is None` (E711), split one-line compound statements
    (E701), `type(x) == Y`→`isinstance` or `is` (E721), move stray imports to top
    or `# noqa: E402` where a late import is intentional (e.g. after
    `sys.path.append`).
- CI gate: `ruff check .` exits 0.

### 3. GitHub Actions CI — `.github/workflows/ci.yml`

- Triggers: `push` and `pull_request`.
- Job: `ubuntu-latest`, `actions/setup-python@v5` with Python 3.12, pip cache.
- Install: `pip install -r requirements.txt --extra-index-url
  https://download.pytorch.org/whl/cpu` then `pip install ruff pytest` (or rely on
  them being pinned in requirements).
- Steps: `ruff check .` then `pytest`. Weight/data-dependent tests skip (absent in
  CI), so the run is green.
- Add the status badge to the top of `README.md` (replacing the Phase 2/3 badge
  placeholder).

### 4. `CLAUDE.md`

- **Entry points** table (the 6 scripts + `app.py`/`gradio_ui.py`/`launch_ui.py`)
  with one-line roles.
- **Architecture note:** `models/` is canonical (GroupNorm, matches bundled
  weights); `resnext/` for the UI ensemble; `utils/` helpers (metrics,
  reproducibility, run_logger).
- **Known pitfalls:** set `PYTHONUTF8=1`; pass `--num_workers 0` on Windows;
  CPU-only → checkpoint loads need `map_location`; `weights/` and `dataset/` are
  gitignored (download via `weights/download_weights.sh` / the OneDrive link;
  generate the smoke set with `scripts/make_smoke_dataset.py`).
- **Testing:** `pytest` (CI), or run any `test_*.py` standalone.

## Verification

- `pytest` green locally (with weights → detector tests run; without → they skip,
  zero failures). Confirm by temporarily renaming `weights/` and re-running.
- `ruff check .` exits 0 across the repo (notebooks excluded).
- The two real bugs are fixed: `utils/misc.py` imports torch; `resnext/model.py`
  has no undefined `model_urls`.
- `.github/workflows/ci.yml` is valid YAML and references real steps; the README
  badge URL points at the repo's Actions.
- Existing behavior unbroken: the full entry-point smoke matrix still runs (spot
  re-check `eval.py` + `synermix_pretrain.py` on `dataset/_smoke`), no `nan`.

## Risks

- **Star-import `__all__`** could omit a name some module imports, causing an
  ImportError. Mitigated by re-running the smoke matrix + `pytest` after the lint
  pass, and preferring `# noqa: F403` over an incomplete `__all__` when unsure.
- **Auto-fix removing a "used via star-import" name** — ruff's `F401` autofix may
  remove an import that is actually re-exported. Review the `--fix` diff before
  committing; add to `__all__` / `# noqa: F401` for intentional re-exports.
- **CI torch install time/flakiness** — CPU wheels are large; use pip caching and
  the explicit CPU index. Acceptable for a portfolio repo.
- **CI first run** only happens after master is pushed to origin (currently
  unpushed); the workflow lands now and goes green on the next push.

## Out of scope

Hugging Face Spaces deploy (separate deferred mini-phase), coverage gates,
multi-OS/multi-Python CI matrices, pre-commit hooks, type-checking (mypy).
