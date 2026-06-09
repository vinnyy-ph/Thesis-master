# T-GD Portfolio Sprint — Design

**Date:** 2026-06-10
**Status:** Approved
**Goal:** Turn the T-GD + SynerMix thesis repo into a portfolio-grade project: an impressive GitHub repo that tells the SynerMix story, plus a live hosted demo. Thesis is complete; no research deadlines apply.

## Context

The repo is a fork of T-GD (Transferable GAN-generated Images Detection Framework, ICML 2020) extended with the author's thesis contribution: **SynerMix** — a synergistic mixing augmentation combining intra-class feature mixing with inter-class CutMix under dynamic beta scheduling, integrated into the pretraining pipeline.

A multi-agent audit (2026-06-10) found:

- `synermix_pretrain.py:571-597` contains an orphaned, unreachable training loop — the headline contribution cannot run end-to-end.
- Two competing EfficientNet packages exist (`models/` vs `EfficientNet/`); `transfer.py` imports the latter while other scripts use the former — silent divergence risk. `resnext/` is unused.
- `transfer.py` logs `arc.avg` (AUROC meter) that is never updated, and applies `0*loss_sp` in one path vs `sp_gamma*loss_sp` in another.
- Reproducibility is near-zero: no seeds, no result export, empty `log/log.txt`, unpinned `requirements.txt`.
- Metrics are thin: loss, top-1, AUROC only.
- `gradio_ui.py` triplicates model-loading code; no explainability; silent `weights_only=False` fallback; `launch_ui.py` crashes on Windows cp1252 consoles (emoji prints).
- `README.md` is still the upstream T-GD paper README — SynerMix is undocumented.
- 9 modified files sit uncommitted (device-handling hardening, mkdir fixes).

## Decisions

- **Audience:** portfolio visitors — recruiters skimming the README and ML-savvy reviewers reading code.
- **Deliverables:** (1) clean, documented repo with real results table; (2) live demo on Hugging Face Spaces.
- **Approach:** phased sprint ordered by visitor visibility; each phase ships standalone value.
- **Out of scope:** ablation re-runs, model retraining, domain-robustness suite, hosted experiment tracking (wandb/MLflow). Revisit only on request.

## Phase 1 — Fix the core

Make the codebase honest: everything importable runs, nothing diverges silently.

1. **Triage uncommitted work.** Review the 9-file diff; commit as coherent, separately-revertable commits.
2. **Restructure `synermix_pretrain.py`.** Move the orphaned loop (571-597) into a `main()` entered via `if __name__ == "__main__"`. The script must run end-to-end on a tiny synthetic dataset.
3. **Consolidate model packages.** All imports point at `models/`; delete `EfficientNet/` (duplicate package) and `resnext/` (unused). Check `gradio_ui.py`, `quick_start.py`, notebooks for stragglers before deleting.
4. **Fix `transfer.py` bugs.** Update the `arc` meter where AUROC is computed, or remove it from logging; reconcile the `0*loss_sp` vs `sp_gamma*loss_sp` inconsistency (decide intended behavior from thesis text and document it).
5. **Strip debug prints** (`synermix_pretrain.py:259, 395, 402, 510`) and dead imports/schedulers.

**Verification:** `test_synermix.py` passes; each entry point (`pretrain.py`, `synermix_pretrain.py`, `transfer.py`, `eval.py`, `quick_start.py`) smoke-runs on CPU with a tiny ImageFolder fixture.

## Phase 2 — Story layer (reproducibility + README)

1. **`utils/reproducibility.py`** — `set_seeds(seed)` covering `random`, `numpy`, `torch` (+CUDA); called from every entry point; `--seed` added to options and saved with checkpoints.
2. **`utils/metrics.py`** — `compute_metrics(y_true, y_score)` returning accuracy, precision, recall, F1, average precision, AUROC, confusion matrix. Binary-class assertion before AUROC `[:,1]` indexing. Wired into `eval.py`, `quick_start.py`, and validation loops.
3. **Results export.** Per-epoch and final metrics written to `runs/<timestamp>/metrics.json` (+ CSV); the full options namespace serialized alongside checkpoints.
4. **README rewrite.** Thesis title and author; SynerMix method summary (what intra-class mixing + inter-class CutMix + beta scheduling each contribute); results table populated from a real eval run on the bundled weights; quickstart matching this repo (Windows-friendly); upstream T-GD paper credited with its citation.
5. **Pin `requirements.txt`** to the working venv versions.

**Verification:** real `eval.py` run on existing weights + test set produces the README table numbers. If the local `dataset/` lacks the needed test split, fall back to thesis-reported numbers, labeled as such.

## Phase 3 — Demo layer

1. **`load_detector()` helper** — single function owning checkpoint load + SynerMix state-dict remapping; replaces the 3 copies in `gradio_ui.py`. Explicit, logged `weights_only` handling instead of silent unsafe fallback.
2. **GradCAM overlay** — heatmap rendered beside the verdict (model already caches feature maps in `models/model.py`). Doubles as a thesis-quality figure.
3. **Windows console fix** — UTF-8 stdout reconfigure at the top of `launch_ui.py` (already crashes on cp1252; reproduced 2026-06-10).
4. **Input validation** — reject corrupt/unsupported files with friendly errors; sanitize exception text shown to users.
5. **Hugging Face Spaces deploy** — `app.py` wrapper, weights via HF Hub (or Git LFS), live link badged in README.

**Verification:** local UI exercised (upload → verdict → GradCAM); then the live Space checked end-to-end.

## Phase 4 — Trust layer

1. **pytest conversion** of `test_synermix.py`; new tests for `compute_metrics` and `load_detector`.
2. **GitHub Actions** — pytest + ruff on push; badge in README.
3. **CLAUDE.md** — entry points, script roles, known pitfalls.

**Verification:** CI green on GitHub.

## Error handling principles

- Entry points fail fast with actionable messages (missing dataset dir, missing weights) instead of deep stack traces.
- Checkpoint loading errors name the expected file and the weight-download step.
- The UI never shows raw tracebacks to visitors.

## Risks

- **Deleting `EfficientNet/`** may break unnoticed imports — mitigated by repo-wide import grep + smoke runs before deletion.
- **`transfer.py` loss_sp reconciliation** changes training behavior — document the chosen semantics; no retraining required since shipped weights stay frozen.
- **HF Spaces resource limits** (CPU-only, 16 GB) — EfficientNet inference is light; ensemble mode may need a model-count cap on the Space.
