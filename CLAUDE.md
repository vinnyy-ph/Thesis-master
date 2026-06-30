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
