# Phase 3: Demo Layer (local) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A clean local Gradio demo (image → real/fake verdict + confidence + GradCAM heatmap), a single shared `detector.py` helper that removes the triplicated checkpoint-load code, and a Windows console fix.

**Architecture:** A new dependency-free-ish `detector.py` owns checkpoint loading (SynerMix remap + logged `weights_only` fallback), preprocessing, prediction, and a hand-rolled GradCAM (hooks on EfficientNet `_conv_head`). A new `app.py` is the focused showcase; the existing `gradio_ui.py` is refactored to load via `detector.load_detector`. `launch_ui.py` and `app.py` reconfigure stdout to UTF-8.

**Tech Stack:** Python 3.12 (`.\venv\Scripts\python.exe`), PyTorch 2.12.0+cpu, torchvision, gradio 6.14.0, matplotlib 3.10.9 (already a dep), PIL, numpy. Windows / PowerShell.

## Global Constraints

- ALWAYS prefix Python runs with `$env:PYTHONUTF8='1'; ` (cp1252 console crashes on Unicode).
- This machine is CPU-only; checkpoint loads pass `map_location`. Smoke images live in `dataset/_smoke`.
- Run from repo root `C:\Users\Vincent\Documents\GitHub\Portfolio\Thesis-master` with `.\venv\Scripts\python.exe`.
- pytest is NOT installed — tests are standalone `assert` scripts run via `python <test>.py` (exit 0 = pass). Do NOT add pytest (Phase 4) and do NOT add `grad-cam` (GradCAM is hand-rolled).
- Canonical model package is `models/` (GroupNorm EfficientNet); `resnext/` for ResNeXt.
- Class index convention: `ImageFolder` sorts classes alphabetically → `fake`=0, `real`=1.
- Do NOT launch the Gradio server in any automated verification (it blocks). Verify by calling functions/callbacks directly.

---

### Task 1: `detector.py` — load_detector + preprocess + predict

**Files:**
- Create: `detector.py`
- Create: `test_detector.py`

**Interfaces:**
- Produces: `load_detector(checkpoint_path, arch='efficientnet-b0', num_classes=2, device=None) -> torch.nn.Module`; `preprocess(pil_image, size=256) -> Tensor(1,3,size,size)`; `predict(model, pil_image, device=None, size=256) -> {'Real': float, 'Fake': float}`. (Task 2 adds `gradcam` to the same file.)

- [ ] **Step 1: Write the failing test**

Create `test_detector.py`:

```python
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
```

- [ ] **Step 2: Run it, confirm it fails on import**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_detector.py
```
Expected: `ModuleNotFoundError: No module named 'detector'`.

- [ ] **Step 3: Implement `detector.py` (load/preprocess/predict)**

Create `detector.py`:

```python
"""Detector loading + inference + GradCAM for the GAN-image detector demo.

Single source of truth for checkpoint loading (SynerMix state-dict remapping
and explicit, logged weights_only handling), preprocessing, prediction, and a
hand-rolled GradCAM overlay (added in Task 2). No Gradio import here, so these
functions stay unit-testable.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from models import EfficientNet
from resnext import resnext50_32x4d

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# ImageFolder sorts classes alphabetically: fake=0, real=1
_FAKE_IDX, _REAL_IDX = 0, 1


def _resolve_device(device):
    if device is not None:
        return device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _remap_synermix(state_dict):
    """Remap a SynerMix-saved state dict to the plain EfficientNet layout.

    SynerMix wraps the backbone, so keys gain a 'base_model.' prefix and the
    head is named 'classifier.*' instead of '_fc.*'. No-op for plain dicts.
    """
    if not any(k.startswith('base_model.') for k in state_dict):
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            remapped[k[len('base_model.'):]] = v
        elif k == 'classifier.weight':
            remapped['_fc.weight'] = v
        elif k == 'classifier.bias':
            remapped['_fc.bias'] = v
        else:
            remapped[k] = v
    return remapped


def load_detector(checkpoint_path, arch='efficientnet-b0', num_classes=2, device=None):
    """Load a detector checkpoint into the canonical model, eval-ready.

    Tries torch.load(weights_only=True); on failure logs a warning and falls
    back to weights_only=False (the bundled checkpoints are trusted local
    files). Applies the SynerMix remap before load_state_dict.
    """
    device = _resolve_device(device)
    if arch.startswith('efficientnet'):
        model = EfficientNet.from_name(arch, num_classes=num_classes)
    elif arch.startswith('resnext'):
        model = resnext50_32x4d(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001 - trusted local file fallback
        logger.warning(
            "weights_only=True load failed for trusted local checkpoint %s (%s); "
            "falling back to weights_only=False", checkpoint_path, e)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    state_dict = _remap_synermix(state_dict)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def preprocess(pil_image, size=256):
    """RGB PIL -> normalized (1,3,size,size) tensor (ImageNet stats)."""
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return tfm(pil_image.convert('RGB')).unsqueeze(0)


def predict(model, pil_image, device=None, size=256):
    """Return {'Real': p, 'Fake': p} softmax probabilities for an image."""
    device = _resolve_device(device)
    x = preprocess(pil_image, size).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    return {'Real': float(probs[_REAL_IDX]), 'Fake': float(probs[_FAKE_IDX])}
```

- [ ] **Step 4: Run the test, confirm pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_detector.py
```
Expected: `detector load/predict tests passed.`, exit 0. (~30s on CPU: builds EfficientNet-b0 and loads a 31 MB checkpoint twice.)

- [ ] **Step 5: Commit**

```powershell
git add detector.py test_detector.py
git commit -m "feat: add detector.load_detector/preprocess/predict with test"
```

---

### Task 2: `detector.py` — hand-rolled GradCAM

**Files:**
- Modify: `detector.py`
- Modify: `test_detector.py`

**Interfaces:**
- Consumes: `load_detector`, `preprocess` from Task 1.
- Produces: `gradcam(model, pil_image, device=None, size=256, alpha=0.5) -> PIL.Image` (size×size overlay).

- [ ] **Step 1: Add the failing test**

Append to `test_detector.py` (add imports `from detector import gradcam` at top; add the test and call it in `__main__`):

```python
def test_gradcam_overlay():
    model = load_detector(CKPT, device=CPU)
    ov = gradcam(model, _rand_image(), device=CPU, size=64)
    assert isinstance(ov, Image.Image)
    assert ov.size == (64, 64)
    assert np.isfinite(np.asarray(ov, dtype=np.float32)).all()
    # hooks must be removed after the call
    assert len(model._conv_head._forward_hooks) == 0
    assert len(model._conv_head._full_backward_hooks) == 0
```
And in `__main__`, add `test_gradcam_overlay()` before the print, and update the print to `"detector load/predict/gradcam tests passed."`.

- [ ] **Step 2: Run it, confirm it fails**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_detector.py
```
Expected: `ImportError: cannot import name 'gradcam'`.

- [ ] **Step 3: Implement `gradcam` (+ `_colorize`) in `detector.py`**

Append to `detector.py`:

```python
def _colorize(cam_norm):
    """Map a [0,1] CAM array to an RGB 'jet' heatmap (uint8, HxWx3)."""
    from matplotlib import colormaps
    rgba = colormaps['jet'](cam_norm)  # (H,W,4) float in [0,1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def gradcam(model, pil_image, device=None, size=256, alpha=0.5):
    """Hand-rolled GradCAM overlay on the EfficientNet ``_conv_head`` layer.

    Forwards the image with gradients enabled, backprops the top predicted
    class logit, weights the captured activations by their pooled gradients,
    ReLU + normalize, then alpha-blends a colormap over the resized input.
    Returns a (size x size) PIL.Image. Hooks are always removed.
    """
    device = _resolve_device(device)
    x = preprocess(pil_image, size).to(device)

    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, output):
        activations['value'] = output.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    target = model._conv_head
    h_fwd = target.register_forward_hook(fwd_hook)
    h_bwd = target.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
        logits[0, pred].backward()
        acts = activations['value'][0]            # (C, h, w)
        grads = gradients['value'][0]             # (C, h, w)
        weights = grads.mean(dim=(1, 2))          # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)
    finally:
        h_fwd.remove()
        h_bwd.remove()

    cam = cam.cpu().numpy().astype(np.float32)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BILINEAR)
    cam_norm = np.asarray(cam_img, dtype=np.float32) / 255.0

    heatmap = _colorize(cam_norm).astype(np.float32)                       # (size,size,3)
    base = np.asarray(pil_image.convert('RGB').resize((size, size)), dtype=np.float32)
    overlay = (alpha * heatmap + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
```

- [ ] **Step 4: Run the test, confirm pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_detector.py
```
Expected: `detector load/predict/gradcam tests passed.`, exit 0.

- [ ] **Step 5: Commit**

```powershell
git add detector.py test_detector.py
git commit -m "feat: add hand-rolled GradCAM overlay to detector"
```

---

### Task 3: `app.py` — clean Gradio showcase

**Files:**
- Create: `app.py`
- Modify: `README.md` (add a short "Demo" section)

**Interfaces:**
- Consumes: `detector.load_detector`, `detector.predict`, `detector.gradcam`.
- Produces: `analyze(pil_image, model_name) -> (label_dict, overlay_image_or_None, status_markdown)`; `build_demo() -> gr.Blocks`; `available_efficientnet_weights() -> list[str]`. The module must be importable without launching a server.

- [ ] **Step 1: Implement `app.py`**

Create `app.py`:

```python
"""Local Gradio demo: upload an image -> real/fake verdict + GradCAM heatmap."""
import glob
import os
import sys
import traceback

import gradio as gr
import torch
from PIL import Image, UnidentifiedImageError

from detector import gradcam, load_detector, predict

# UTF-8 console so any Unicode prints don't crash a cp1252 (Windows) terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_WEIGHT = os.path.join('weights', 't-gd', 'efficientnet', 'star_to_style2.pth.tar')

_MODEL_CACHE = {}


def available_efficientnet_weights():
    """All EfficientNet checkpoints under weights/ (pre-train + t-gd)."""
    paths = sorted(glob.glob(os.path.join('weights', '**', 'efficientnet', '*.pth.tar'),
                             recursive=True))
    return paths


def _get_model(path):
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = load_detector(path, arch='efficientnet-b0', device=DEVICE)
    return _MODEL_CACHE[path]


def analyze(pil_image, model_name):
    """Return (label dict, overlay image, status markdown). Never raises to UI."""
    if pil_image is None:
        return None, None, "**Please upload an image.**"
    if not isinstance(pil_image, Image.Image):
        try:
            pil_image = Image.open(pil_image)
        except (UnidentifiedImageError, OSError, ValueError):
            return None, None, "**Could not read that file — please upload a valid PNG/JPG image.**"
    weight = model_name or DEFAULT_WEIGHT
    try:
        model = _get_model(weight)
        probs = predict(model, pil_image, device=DEVICE)
        overlay = gradcam(model, pil_image, device=DEVICE)
        verdict = 'Fake (GAN-generated)' if probs['Fake'] >= probs['Real'] else 'Real'
        status = f"### Verdict: {verdict}\nModel: `{os.path.basename(weight)}`"
        return probs, overlay, status
    except Exception:  # noqa: BLE001 - keep the UI friendly; log server-side
        traceback.print_exc()
        return None, None, "**Something went wrong analyzing that image. Check the server log.**"


def build_demo():
    weights = available_efficientnet_weights()
    default = DEFAULT_WEIGHT if DEFAULT_WEIGHT in weights else (weights[0] if weights else DEFAULT_WEIGHT)
    with gr.Blocks(title="SynerMix GAN-Image Detector") as demo:
        gr.Markdown("# SynerMix GAN-Image Detector\nUpload a face image — the detector "
                    "predicts real vs. GAN-generated and shows a GradCAM heatmap of where it looked.")
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Image")
                model_dd = gr.Dropdown(choices=weights, value=default, label="Detector weights")
                run = gr.Button("Analyze", variant="primary")
            with gr.Column():
                status_out = gr.Markdown()
                label_out = gr.Label(num_top_classes=2, label="Probabilities")
                cam_out = gr.Image(type="pil", label="GradCAM")
        run.click(analyze, inputs=[image_in, model_dd], outputs=[label_out, cam_out, status_out])
    return demo


if __name__ == '__main__':
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Verify the module imports and the callback works (do NOT launch the server)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import numpy as np; from PIL import Image; import app; img=Image.fromarray(np.random.RandomState(1).randint(0,256,(64,64,3),dtype=np.uint8)); probs,ov,st=app.analyze(img, app.DEFAULT_WEIGHT); print('probs', probs); print('overlay', ov.size if ov else None); print('weights found', len(app.available_efficientnet_weights()))"
```
Expected: prints a `probs` dict with Real/Fake, an `overlay` size tuple, and `weights found` ≥ 1. No traceback.

- [ ] **Step 3: Verify input validation returns a friendly message (no exception)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import app; print(app.analyze(None, None))"
```
Expected: `(None, None, '**Please upload an image.**')`.

- [ ] **Step 4: Add a "Demo" section to `README.md`**

Insert a short section after the Quickstart (before "Entry points"):

```markdown
## Demo (local)

A focused Gradio demo — upload an image, get a real/fake verdict plus a GradCAM
heatmap of where the detector looked:

```powershell
$env:PYTHONUTF8='1'; python app.py   # then open http://localhost:7860
```
```

- [ ] **Step 5: Commit**

```powershell
git add app.py README.md
git commit -m "feat: add clean local Gradio demo with GradCAM (app.py)"
```

---

### Task 4: Dedup `gradio_ui.py` model loading

`gradio_ui.py` builds the model and loads+remaps the checkpoint in 3 near-identical places: ~lines 105–162 (`load_model`), ~261–316 (`analyze_with_all_models`), ~603–658 (`analyze_multiple_images_with_all_models`). Replace each with a single `detector.load_detector(...)` call.

**Files:**
- Modify: `gradio_ui.py`

**Interfaces:**
- Consumes: `detector.load_detector(path, arch=..., device=...)`.

- [ ] **Step 1: Read the 3 load sites**

Open `gradio_ui.py` and locate each block: the `model = EfficientNet.from_name(...)/resnext50_32x4d(...)` build (arch chosen from `model_info['arch']`), the `torch.load` weights_only try/fallback, the `state_dict = checkpoint['state_dict'] ...`, and the `base_model.`/`classifier.*` remap, ending at `model.load_state_dict(state_dict)`.

- [ ] **Step 2: Add the import**

At the top of `gradio_ui.py` with the other imports, add:
```python
from detector import load_detector
```

- [ ] **Step 3: Replace each of the 3 blocks**

In each of the 3 sites, replace the whole sequence (arch build → torch.load → state_dict extraction → SynerMix remap → `model.load_state_dict(state_dict)` → `model.to(device)` → `model.eval()`) with:
```python
            arch = 'efficientnet-b0' if model_info['arch'] == 'efficientnet' else 'resnext50_32x4d'
            model = load_detector(model_info['path'], arch=arch, device=self.device)
```
(Keep the surrounding code — the `model_info` lookup before, and the assignment to `self.current_model`/use of `model` after. `load_detector` already returns an eval-mode model on `self.device`, so drop the now-redundant `.to()/.eval()` lines. Leave ensemble/multi-image logic, dropdowns, and outputs untouched.)

- [ ] **Step 4: Verify the dedup without importing (importing may launch the server)**

First confirm `gradio_ui.py`'s `.launch()` is under an `if __name__ == '__main__':` guard. If it is, importing is safe; if NOT, do not import it in verification (it would block) — rely on the source check below either way, which does not import:
```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "src=open('gradio_ui.py',encoding='utf-8').read(); assert 'from detector import load_detector' in src, 'import missing'; assert 'load_detector(' in src, 'helper not called'; assert src.count('base_model.') == 0, 'remap still duplicated in gradio_ui'; print('dedup verified: load_detector wired, remap centralized in detector.py')"
```
Expected: `dedup verified: ...`. (After dedup, the `base_model.` remap literal lives only in `detector.py`, so its count in `gradio_ui.py` is 0.)

Then byte-compile to catch syntax errors from the edits (does not execute module-level code):
```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m py_compile gradio_ui.py; echo "compile EXIT=$LASTEXITCODE"
```
Expected: `compile EXIT=0`.

- [ ] **Step 5: Commit**

```powershell
git add gradio_ui.py
git commit -m "refactor: load gradio_ui models via shared detector.load_detector"
```

---

### Task 5: Windows console fix in `launch_ui.py`

`launch_ui.py` prints emoji that crash a cp1252 console with `UnicodeEncodeError`.

**Files:**
- Modify: `launch_ui.py`

- [ ] **Step 1: Add a UTF-8 stdout reconfigure at the very top**

In `launch_ui.py`, immediately after the existing imports (ensure `import sys` is present), add:
```python
# UTF-8 console so emoji prints don't crash a cp1252 (Windows) terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
```

- [ ] **Step 2: Verify it imports and prints cleanly**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import sys; sys.argv=['launch_ui.py']; import importlib.util as u; print('launch_ui present:', u.find_spec('launch_ui') is not None)"
```
Then exercise the reconfigure + an emoji print directly (this is the crash that the fix prevents):
```powershell
.\venv\Scripts\python.exe -c "import sys; [s.reconfigure(encoding='utf-8') for s in (sys.stdout,sys.stderr) if hasattr(s,'reconfigure')]; print('console OK 🚀✅')"
```
Expected: prints `console OK 🚀✅` with NO `UnicodeEncodeError` (note: do NOT set `$env:PYTHONUTF8='1'` for this second check — the point is the in-code reconfigure works even on a cp1252 default console).

- [ ] **Step 3: Commit**

```powershell
git add launch_ui.py
git commit -m "fix: reconfigure stdout to UTF-8 in launch_ui for Windows consoles"
```

---

### Task 6: Final verification

**Files:** none (verification only).

- [ ] **Step 1: detector tests pass**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_detector.py
```
Expected: `detector load/predict/gradcam tests passed.`, exit 0.

- [ ] **Step 2: app.py callback end-to-end on a real smoke image**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import app; from PIL import Image; img=Image.open('dataset/_smoke/test/fake/fake_0.png'); probs,ov,st=app.analyze(img, None); print('OK', probs, ov.size, st[:40])"
```
Expected: a probs dict, an overlay size, a verdict string; no traceback, exit 0.

- [ ] **Step 3: gradio_ui + launch_ui byte-compile (avoid import — they may launch the server)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -m py_compile gradio_ui.py launch_ui.py app.py detector.py; echo "compile EXIT=$LASTEXITCODE"
```
Expected: `compile EXIT=0` (no syntax errors). Importing these is avoided because `gradio_ui.py`/`launch_ui.py` may call `.launch()` / run setup at import time.

- [ ] **Step 4: Pre-existing tests still pass (no regression)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: `All tests passed successfully!`, exit 0.

- [ ] **Step 5: Clean tree**

```powershell
git status --short
```
Expected: only `?? PORTFOLIO_ITEMS.md` untracked. Report any deviation.
