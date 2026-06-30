# Phase 3 — Demo Layer (local)

**Date:** 2026-06-30
**Status:** Approved
**Parent spec:** `2026-06-10-tgd-portfolio-sprint-design.md` (Phase 3)
**Prereq:** Phases 1–2 merged to master.

## Goal

A clean, focused local demo of the SynerMix detector: upload an image → real/fake
verdict + confidence + a GradCAM heatmap showing where the model looked. Plus a
single shared model-loading helper that removes the triplicated load code, and a
Windows console fix. Hugging Face Spaces hosting is **deferred to a later
mini-phase** (decided in brainstorming, 2026-06-30).

## Decisions (from brainstorming, 2026-06-30)

- **Hosting deferred** — local demo only this phase. No HF Spaces, no weight
  hosting, no `app.py`-on-Spaces concerns yet.
- **New `app.py` showcase + dedup the old UI** — build a focused new demo;
  extract a shared `load_detector()` and refactor `gradio_ui.py`'s 3 duplicate
  load blocks to use it (its ensemble/multi-image features stay otherwise).
- **GradCAM hand-rolled** — forward+backward hooks on the EfficientNet
  `_conv_head` layer; no new dependency (keeps `requirements.txt` curated).
- **Single EfficientNet detector** for the showcase (light, CPU-friendly, matches
  the bundled GroupNorm weights), with an optional dropdown of available
  EfficientNet checkpoints.

## Context (from a 2026-06-30 code audit)

- `gradio_ui.py` (gr.Blocks, ~1000 lines, ensemble + multi-image) duplicates the
  checkpoint-load + SynerMix-remap logic in 3 places: lines ~110–138 (`load_model`),
  ~266–294 (`analyze_with_all_models`), ~609–636 (`analyze_multiple_images_with_all_models`).
  Each tries `torch.load(..., weights_only=True)` then falls back to
  `weights_only=False` on exception, then remaps the SynerMix state dict (strip a
  `base_model.` prefix; rename `classifier.weight`/`classifier.bias` →
  `_fc.weight`/`_fc.bias`).
- GradCAM target: `models/model.py` EfficientNet `_conv_head` (followed by
  GroupNorm `_gn1`, 1280 channels) — the last conv feature map before pooling.
  `forward()` returns logits only; `extract_features()` exists but is unused.
- gradio **6.14.0** is installed; `grad-cam` is NOT installed (and won't be added).
- `launch_ui.py` prints emoji (✅❌🚀🌐🔄👋) that crash the Windows cp1252 console
  with `UnicodeEncodeError`.
- Weights are large (~1.1 GB total; EfficientNet ckpt ~31 MB each, ResNeXt ~175 MB
  each) — irrelevant this phase since hosting is deferred; the demo loads local
  checkpoints.

## Components

### 1. `detector.py` (new, repo root)

Single source of truth for loading and explaining the detector. Pure functions
(no Gradio import) so they are unit-testable.

- `load_detector(checkpoint_path, arch='efficientnet-b0', num_classes=2, device=None) -> torch.nn.Module`
  - `device` defaults to CUDA-if-available else CPU.
  - Builds the arch (`EfficientNet.from_name` for efficientnet; `resnext50_32x4d`
    for resnext) from `models/` / `resnext/`.
  - Loads the checkpoint with **explicit, logged** safety handling: try
    `torch.load(path, map_location=device, weights_only=True)`; on failure, log a
    clear one-line warning ("falling back to weights_only=False for trusted local
    checkpoint <path>") and retry with `weights_only=False`. Always pass
    `map_location=device` (CPU-only box).
  - Applies the SynerMix state-dict remap: strip a leading `base_model.` from keys
    if present; rename `classifier.weight`/`classifier.bias` →
    `_fc.weight`/`_fc.bias` if present. Then `load_state_dict`.
  - Returns `model.to(device).eval()`.
- `preprocess(pil_image, size=256) -> torch.Tensor` — Resize(size) + ToTensor +
  Normalize(ImageNet mean/std), returns a `(1, 3, H, W)` batch. (Same normalization
  as `eval.py`; `size` configurable, default 256 — not load-bearing since
  EfficientNet pools adaptively.)
- `predict(model, pil_image, device, size=256) -> dict` — returns
  `{'Real': float, 'Fake': float}` from `softmax`. Class index convention follows
  the training data (`ImageFolder` sorts `fake`=0, `real`=1); the dict labels map
  accordingly and this convention is documented in the function docstring.
- `gradcam(model, pil_image, device, size=256, alpha=0.5) -> PIL.Image` — hand-rolled:
  register a forward hook (capture `_conv_head` activations) and a full backward
  hook (capture gradients) on `model._conv_head`; forward the preprocessed image
  (gradients ENABLED — not under `torch.no_grad`); backprop the top predicted
  class logit; weight activations by global-average-pooled gradients; ReLU; upsample
  to the input size; min-max normalize; apply a colormap and alpha-blend over the
  original image. Remove the hooks before returning. Returns an overlay
  `PIL.Image` the same size as the (resized) input.

### 2. `app.py` (new, repo root) — the showcase demo

- A `gr.Blocks` UI: one image upload input; outputs = a verdict label
  (Real/Fake + confidence), a `gr.Label`/bar of the two class probabilities, and
  the GradCAM overlay image, laid out side by side.
- A single default detector: `weights/t-gd/efficientnet/star_to_style2.pth.tar`,
  loaded once at startup via `load_detector`. An optional `gr.Dropdown` lists the
  available EfficientNet checkpoints under `weights/` (pre-train + t-gd); switching
  reloads via `load_detector` (cached by path to avoid reloading the same one).
- The analyze callback: validate the upload (see §3) → `predict` → `gradcam` →
  return verdict + probabilities + overlay.
- UTF-8 stdout reconfigure at the very top (see §5). `python app.py` launches it
  (default `server_port=7860`); prints the local URL.

### 3. Input validation + friendly errors (in `app.py`)

- Reject `None`/empty uploads, non-image files, and unsupported/corrupt images
  (catch PIL `UnidentifiedImageError`/`OSError`) with a clear user-facing message
  (e.g. "Please upload a valid PNG/JPG image.").
- Wrap the analyze callback so any unexpected exception returns a sanitized,
  friendly message to the UI instead of a raw traceback; log the full error
  server-side.

### 4. `gradio_ui.py` dedup

- Replace each of the 3 load+remap blocks (~110–138, ~266–294, ~609–636) with a
  call to `detector.load_detector(checkpoint_path, arch=..., device=...)`.
- Behavior otherwise unchanged (ensemble, multi-image, dropdown, outputs). The
  `weights_only` fallback and SynerMix remap now live solely in `detector.py`.

### 5. Windows console fix

- At the top of `launch_ui.py` AND `app.py`, reconfigure stdout/stderr to UTF-8
  defensively:
  ```python
  import sys
  for _stream in (sys.stdout, sys.stderr):
      if hasattr(_stream, "reconfigure"):
          try:
              _stream.reconfigure(encoding="utf-8")
          except Exception:
              pass
  ```
  This lets the existing emoji prints work on a cp1252 console without rewriting
  every print. (Reconfigure is preferred over stripping emoji — fewer edits, keeps
  the friendly output.)

## Testing

`test_detector.py` (plain `assert` script run with `python test_detector.py`;
pytest is Phase 4):
- `load_detector` on a bundled EfficientNet checkpoint (e.g.
  `weights/pre-train/efficientnet/stargan.pth.tar`) returns a module in `eval()`
  mode; a forward pass on a random `(1,3,64,64)` tensor yields a `(1,2)` output.
- `predict` on a synthetic PIL image returns a dict whose two values are finite
  and sum to ~1.0.
- `gradcam` returns a `PIL.Image` matching the resized input size, with finite
  pixel values (no NaN), and the model's params have no leftover grad hooks
  afterward (hooks removed).
- Use `$env:PYTHONUTF8='1'`, `map_location`-safe loading; runs on CPU.

## Verification

- `python app.py` launches; uploading a `dataset/_smoke` image returns a verdict,
  probabilities, and a GradCAM overlay; an invalid file shows the friendly error,
  not a traceback.
- `gradio_ui.py` still constructs its models through `detector.load_detector`
  (smoke-import / instantiate check; no behavior regression in model loading).
- `launch_ui.py` runs its dependency/weights checks and prints cleanly on the
  Windows console (no `UnicodeEncodeError`).
- `test_detector.py` passes (exit 0).

## Risks

- **GradCAM on the custom GroupNorm EfficientNet** — the hook must target the
  right module (`_conv_head`); mitigated by the unit test asserting overlay shape
  and finiteness, and a manual check that the heatmap varies across images.
- **Class-index convention** (fake=0/real=1 from `ImageFolder`) — if mislabeled,
  verdicts invert; documented in `predict` and checked by eyeballing a known image
  during verification.
- **gradio 6.x API drift** — the existing `gradio_ui.py` already targets 6.14.0,
  so `app.py` follows the same idioms; no version change.

## Out of scope

Hugging Face Spaces deploy, weight hosting / HF Hub, ensemble GradCAM, model
retraining, and converting tests to pytest (Phase 4).
