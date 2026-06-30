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
