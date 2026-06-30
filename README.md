# SynerMix <!-- FILL: replace with the official thesis title -->

**Synergistic Mixing Augmentation for Transferable GAN-Image Detection**

**Author:** Vincent Ferrer

<!-- FILL after Phase 3/4: live-demo badge, CI badge -->

Detecting GAN-generated face images with a *transferable* detector, extended with
**SynerMix** — a synergistic mixing augmentation applied during pretraining. This
repository is a fork of **T-GD** (ICML 2020), with SynerMix as the thesis
contribution.

> **Thesis specifics marked `FILL` below** (title, results numbers) are filled in
> after a full run; the code, method, and harness are complete and runnable today.

## What is this

GAN-generated images carry subtle, model-specific fingerprints. T-GD trains an
EfficientNet-b0 binary classifier (real vs. generated) and *transfers* it across
GAN families (PGGAN, StarGAN, StyleGAN, StyleGAN2) using L2-SP self-training, so a
detector trained on one generator adapts to another with little target data. This
repo adds **SynerMix**, a mixing-based augmentation, to the pretraining stage.

## SynerMix — the contribution

SynerMix combines two complementary mixing strategies under a dynamic schedule.
The implementation lives in [`synermix_pretrain.py`](synermix_pretrain.py).

- **Intra-class feature mixing** — weighted blends of feature vectors *within* a
  class enrich the representation without crossing the decision boundary,
  strengthening intra-class compactness.
- **Inter-class CutMix** — CutMix *across* classes with area-corrected, per-sample
  λ (the label weight matches the pasted-patch area), giving honest boundary
  regularization.
- **Dynamic β schedule** — a warm-up (`--synermix_warmup_epochs`) defers mixing
  until the backbone stabilizes; afterward `--synermix_beta` balances the
  intra-class and inter-class objectives.

The goal: tighter class clusters *and* better-calibrated boundaries than CutMix
alone, improving the detector's transferability.

## Results

*Numbers as reported in the thesis.* Regenerate them with `eval.py` after
downloading a real test set (see [Datasets](#datasets)); the metrics pipeline
(`utils/metrics.py`) and per-run export (`runs/<timestamp>/metrics.json`) are
already wired into every entry point.

| Transfer (source → target) | AUROC | Accuracy |
|----------------------------|-------|----------|
| PGGAN → StarGAN            | <!-- FILL --> | <!-- FILL --> |
| PGGAN → StyleGAN           | <!-- FILL --> | <!-- FILL --> |
| PGGAN → StyleGAN2          | <!-- FILL --> | <!-- FILL --> |
| StarGAN → StyleGAN         | <!-- FILL --> | <!-- FILL --> |
| StarGAN → StyleGAN2        | <!-- FILL --> | <!-- FILL --> |
| StyleGAN → StarGAN         | <!-- FILL --> | <!-- FILL --> |
| StyleGAN → StyleGAN2       | <!-- FILL --> | <!-- FILL --> |
| StyleGAN2 → StarGAN        | <!-- FILL --> | <!-- FILL --> |
| StyleGAN2 → StyleGAN       | <!-- FILL --> | <!-- FILL --> |

<!-- FILL: optional — baseline (T-GD without SynerMix) comparison column or row -->

## Quickstart (Windows / PowerShell)

```powershell
# 1. Environment (Python 3.12)
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Weights are under weights/ (or run weights/download_weights.sh)

# 3. Zero-download sanity check: generate a tiny synthetic dataset
$env:PYTHONUTF8='1'; python scripts\make_smoke_dataset.py

# 4. Evaluate the bundled weights on the smoke set
$env:PYTHONUTF8='1'; python quick_start.py --source_dataset dataset/_smoke `
  --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 `
  --resume weights/pre-train/efficientnet/stargan.pth.tar
```

> **Windows notes** (apply to every script):
> - Always set `$env:PYTHONUTF8='1'` — scripts print Unicode that crashes the
>   default cp1252 console.
> - Always pass `--num_workers 0` — the training transforms use
>   `transforms.Lambda`, which Windows DataLoader workers can't pickle.
>
> Runs are reproducible: pass `--seed <n>` (default 7); seeds, config, and
> per-epoch metrics are written to `runs/<timestamp>/`.

## Entry points

| Script | Role |
|--------|------|
| [`synermix_pretrain.py`](synermix_pretrain.py) | **SynerMix pretraining (headline contribution)** |
| [`pretrain.py`](pretrain.py), [`early_stop_pretrain.py`](early_stop_pretrain.py) | Baseline pretraining |
| [`transfer.py`](transfer.py) | L2-SP transfer / self-training to a new GAN |
| [`eval.py`](eval.py) | Full evaluation (source + target) |
| [`quick_start.py`](quick_start.py) | Single-checkpoint quick evaluation |

Each prints accuracy, precision, recall, F1, average precision, and AUROC, and
exports them to `runs/<timestamp>/{config.json, metrics.json, metrics.csv}`.

## Repo layout

| Path | Role |
|------|------|
| `models/` | Canonical EfficientNet backbone (GroupNorm; matches the bundled weights) |
| `resnext/` | ResNeXt backbone (used in the UI's ensemble mode) |
| `utils/` | `metrics.py`, `reproducibility.py`, `run_logger.py`, plus shared helpers |
| `options/` | argparse option groups (base / test / transfer) |
| `scripts/` | `make_smoke_dataset.py` — synthetic CPU smoke fixture |
| `weights/` | Bundled pre-train and T-GD transfer checkpoints |
| `notebooks/` | Archived training notebook |

## Datasets

No dataset ships in the repo (only the synthetic `dataset/_smoke` fixture). The
real test sets — CelebA / CelebA-HQ / FFHQ / LSUN and their GAN counterparts
(PGGAN, StarGAN, StyleGAN, StyleGAN2) — are available from the upstream T-GD
release:
[OneDrive (SKKU)](https://skku0-my.sharepoint.com/:f:/g/personal/byo7000_skku_edu/EoP8mWpbyDhNtIaZ9rBoPWcB5QRsinPBKwr0V18dHsUR8w?e=7oNCXY).
The detection data derives heavily from
[progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans).

## Credits

Built on **T-GD: Transferable GAN-generated Images Detection Framework** —
Hyeonseong Jeon, Youngoh Bang, Junyaup Kim, Simon S. Woo. *ICML 2020.*
Upstream repo: <https://github.com/cutz-j/T-GD>.

```bibtex
@inproceedings{jeon2020tgd,
  title     = {T-GD: Transferable GAN-generated Images Detection Framework},
  author    = {Jeon, Hyeonseong and Bang, Youngoh and Kim, Junyaup and Woo, Simon S.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}
```
