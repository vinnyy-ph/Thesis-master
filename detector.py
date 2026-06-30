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
