# Phase 2 — Story Layer (reproducibility + metrics + README)

**Date:** 2026-06-30
**Status:** Approved
**Parent spec:** `2026-06-10-tgd-portfolio-sprint-design.md` (Phase 2)
**Prereq:** Phase 1 merged to master (every entry point runs end-to-end on CPU).

## Goal

Make the repo *legible and reproducible* to a portfolio visitor: deterministic runs, a real metrics module, machine-readable result export, and a README that tells the SynerMix story instead of the upstream T-GD paper's. This is the layer that turns "the code runs" into "I can see what it does and trust the numbers."

## Decisions (from brainstorming, 2026-06-30)

- **Results table source:** thesis-reported numbers, **clearly labeled** as such; build the metrics + eval harness so a real run can populate measured numbers later. No multi-GB dataset download this phase (no real dataset is present locally — only the synthetic `dataset/_smoke` fixture; all trained weights ARE present under `weights/`).
- **Thesis-specific README content:** scaffold with clearly-marked `<!-- FILL: ... -->` placeholders. Author is **Vincent Ferrer**. The SynerMix method summary is derived from the code. The user fills the thesis title and the results-table cells afterward.
- **Metrics wiring depth:** **deep** — route `eval.py`, `quick_start.py`, AND the validation loops in `pretrain.py` / `transfer.py` / `synermix_pretrain.py` through `metrics.py`, using epoch-level accumulation. This root-fixes the carried-forward per-batch single-class `roc_auc_score` `nan` (sklearn 1.8.0 *returns* `nan`, not `ValueError`) instead of patching guards script-by-script.

## Out of scope (unchanged from parent spec)

Model retraining, ablation re-runs, domain-robustness suite, hosted experiment tracking (wandb/MLflow), the live demo (Phase 3), CI/tests-as-CI (Phase 4). A real dataset download is explicitly deferred — the harness is built ready, but not exercised on real data this phase.

## Components

### 1. `utils/reproducibility.py`

- `set_seeds(seed: int) -> None`: seeds `random`, `numpy`, `torch` (CPU + CUDA all-devices), sets `torch.backends.cudnn.deterministic = True` and `cudnn.benchmark = False` when seeding for reproducibility. Document that this trades a little speed for determinism.
- A `--seed` argument (default **7**, matching the existing `manual_seed` default) added to the options classes used by entry points: `options/base.py`, `options/test.py`, `options/transfer.py`. Where `manual_seed` already exists, `--seed` supersedes it; keep `manual_seed` as a deprecated alias only if removing it would break a referenced flag (check before removing).
- Every entry point (`pretrain.py`, `synermix_pretrain.py`, `transfer.py`, `eval.py`, `quick_start.py`, `early_stop_pretrain.py`) calls `set_seeds(opt.seed)` immediately after options are parsed and before any model/data construction.
- The seed value is recorded in run metadata (see §3) and included in the checkpoint dict saved by training scripts.

### 2. `utils/metrics.py`

- `compute_metrics(y_true, y_score) -> dict` where `y_true` is an array of 0/1 labels and `y_score` is the array of positive-class scores (the `[:, 1]` column of softmax/logits over the 2 classes). Returns a dict with: `accuracy`, `precision`, `recall`, `f1`, `average_precision`, `auroc`, and `confusion_matrix` (as `{tn, fp, fn, tp}`).
- Asserts binary input (exactly the classes {0,1} expected) before indexing the positive column; raises a clear `ValueError` naming the problem if given non-binary data.
- **Single-class safety:** if `y_true` contains only one class, `auroc` and `average_precision` are set to `None` (threshold-free ranking metrics are undefined) while accuracy/precision/recall/f1/confusion-matrix are still computed. Never returns or propagates `nan`. This is the canonical replacement for the ad-hoc per-batch `roc_auc_score` calls.
- A small helper `accumulate` pattern (documented, not necessarily a class): the validation/test loops collect `y_true` and `y_score` across all batches, then call `compute_metrics` **once** at the end of the epoch. This removes per-batch AUROC entirely, so a single-class *batch* can no longer corrupt the epoch metric.
- **Wiring (deep):**
  - `eval.py`, `quick_start.py`: replace the per-batch `arc` AUROC meter with epoch accumulation → one `compute_metrics` call; print the full metric set; return the dict.
  - `pretrain.py`, `transfer.py`, `synermix_pretrain.py` validation/`test()` paths: same accumulation pattern. The training-loss meters (loss, top-1) stay as running averages; only the AUROC computation moves to end-of-epoch via `compute_metrics`. `transfer.py`'s Phase-1 guarded per-batch AUROC is replaced by this accumulation (the guard becomes unnecessary).
  - Keep the printed-line formats close to existing so logs stay familiar; add the new metrics to the printed line and to the logger columns where a logger exists.

### 3. Results export — `utils/run_logger.py`

- On each run, create `runs/<timestamp>/` (timestamp format `YYYYMMDD-HHMMSS`).
- Write `config.json`: the full `vars(opt)` namespace (so a run is reconstructable).
- Write `metrics.json`: `{"final": {...}, "per_epoch": [{...}, ...]}` for training scripts; `{"final": {...}}` for eval scripts. Also write `metrics.csv` (per-epoch rows; one row for eval).
- `runs/` is added to `.gitignore` (generated output, like `log/`). The in-repo visible result is the README table.
- Helper API kept minimal: `start_run(opt) -> run_dir`, `log_epoch(run_dir, epoch, metrics_dict)`, `finalize(run_dir, final_metrics_dict)`. Entry points that train call all three; eval scripts call `start_run` + `finalize`.

### 4. README rewrite

Replace the upstream T-GD README. Structure (Windows-friendly throughout):

1. **Title** — `# SynerMix <!-- FILL: full thesis title -->` and a one-line pitch: detecting GAN-generated face images with a transferable detector, plus the SynerMix augmentation contribution.
2. **Author** — Vincent Ferrer.
3. **Badge row** — placeholder comment for the Phase 3 live-demo badge and Phase 4 CI badge (`<!-- FILL after Phase 3/4 -->`).
4. **What is this** — GAN-image detection in 2–3 sentences; this repo is a fork of T-GD (ICML 2020) extended with SynerMix.
5. **SynerMix — the contribution** — derived from `synermix_pretrain.py`: intra-class feature mixing, inter-class CutMix (area-corrected per-sample λ), and the dynamic β warm-up schedule; one sentence on what each adds. Reference `synermix_pretrain.py` as the headline script.
6. **Results** — a table with rows for the available transfer weights (e.g. `star → style2`), columns AUROC / Accuracy, with `<!-- FILL -->` cells and a caption "Numbers as reported in the thesis." A one-line note on how to regenerate them with `eval.py` once a real test set is downloaded.
7. **Quickstart** — venv activation, `pip install -r requirements.txt`, the `$env:PYTHONUTF8='1'` and `--num_workers 0` Windows gotchas, a `quick_start.py` example against the bundled weights, and the smoke-dataset generator for a zero-download sanity check.
8. **Repo layout** — short table of the key entry points and packages (`models/` canonical, `resnext/` for ensemble, `utils/`, `scripts/`, `notebooks/`).
9. **Credits & citation** — T-GD authors + ICML 2020 BibTeX (preserve from the old README); link to the upstream repo.

Keep the upstream overview image only if the file exists (`image/overview.png`); otherwise drop the reference.

### 5. Pin `requirements.txt`

- Replace the unpinned list with versions pinned to the working venv (`pip freeze` filtered to the actually-imported direct dependencies: torch, torchvision, scikit-learn, scipy, numpy, opencv-python, Pillow, matplotlib, tqdm, torchsummary, progress, and the `git+` gradual-warmup-lr dependency).
- Do not dump the entire `pip freeze` (transitive deps); keep it curated and readable. Pin to `==` the installed versions.

## Verification

- **`metrics.py` unit tests:** known-input cases (perfect separation → AUROC 1.0; single-class input → `auroc is None`, no `nan`; a hand-computed confusion matrix). Run on CPU, fast.
- **`reproducibility.py`:** two seeded `eval.py` (or `quick_start.py`) runs on `dataset/_smoke` with the same `--seed` produce identical `metrics.json`.
- **Smoke runs still pass:** `eval.py` and `quick_start.py` on `dataset/_smoke` (`--test_batch 16`) exit 0 and print the full metric set with no `nan`; the training scripts' validation loops emit the metric set per epoch.
- **`runs/` export:** a smoke run writes `runs/<timestamp>/{config.json, metrics.json, metrics.csv}`.
- **`requirements.txt`:** every pinned version matches the installed venv (`pip show` / `pip freeze` spot-check); no unpinned lines remain except the unavoidable `git+` URL.
- **README:** renders; all non-derivable thesis specifics are `<!-- FILL -->` placeholders (no fabricated numbers); quickstart commands match this repo's actual flags.

## Risks

- **Deep metrics wiring touches 6 scripts** — mitigated by the shared `compute_metrics`/accumulation pattern (one helper, mechanical per-site change) and per-script smoke runs after each edit.
- **cudnn-deterministic** can change throughput and, rarely, disallow an op — acceptable for a CPU-first portfolio repo; documented in `set_seeds`.
- **Pinned versions** could over-constrain a future fresh install — acceptable; the goal is a reproducible snapshot, and the versions are the ones known to work.
