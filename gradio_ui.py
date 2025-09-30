import os
import sys
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import glob
from models import EfficientNet
from resnext import resnext50_32x4d

class T_GD_Interface:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model = None
        self.current_model_info = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Get available models
        self.available_models = self.get_available_models()
        
    def get_available_models(self):
        """Scan weights directory for available models"""
        models = {}
        
        # Pre-trained models
        pretrain_dir = "weights/pre-train"
        if os.path.exists(pretrain_dir):
            for arch in ['efficientnet', 'resnext']:
                arch_path = os.path.join(pretrain_dir, arch)
                if os.path.exists(arch_path):
                    for weight_file in glob.glob(os.path.join(arch_path, "*.pth.tar")):
                        model_name = os.path.basename(weight_file).replace('.pth.tar', '')
                        models[f"Pre-trained: {arch.upper()} - {model_name.upper()}"] = {
                            'path': weight_file,
                            'arch': arch,
                            'type': 'pretrained',
                            'source': model_name,
                            'description': f'Pre-trained {arch} model on {model_name} dataset'
                        }
        
        # Transfer learning models
        tgd_dir = "weights/t-gd"
        if os.path.exists(tgd_dir):
            for arch in ['efficientnet', 'resnext']:
                arch_path = os.path.join(tgd_dir, arch)
                if os.path.exists(arch_path):
                    for weight_file in glob.glob(os.path.join(arch_path, "*.pth.tar")):
                        model_name = os.path.basename(weight_file).replace('.pth.tar', '')
                        source, target = model_name.split('_to_')
                        models[f"T-GD: {arch.upper()} - {source.upper()} → {target.upper()}"] = {
                            'path': weight_file,
                            'arch': arch,
                            'type': 'transfer',
                            'source': source,
                            'target': target,
                            'description': f'Transfer learning from {source} to {target} using {arch}'
                        }
        
        return models
    
    def load_model(self, model_key):
        """Load selected model"""
        if model_key not in self.available_models:
            return "Model not found!"
            
        model_info = self.available_models[model_key]
        
        try:
            # Create model based on architecture
            if model_info['arch'] == 'efficientnet':
                model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
            else:  # resnext
                model = resnext50_32x4d(num_classes=2)
            
            # Load weights (handle PyTorch 2.6+ weights_only behavior safely)
            try:
                # Try safe loading with allowlisted NumPy scalar used in older checkpoints
                try:
                    from numpy.core.multiarray import scalar as numpy_scalar  # type: ignore
                except Exception:
                    numpy_scalar = None

                if hasattr(torch, 'serialization') and numpy_scalar is not None:
                    try:
                        torch.serialization.add_safe_globals([numpy_scalar])
                    except Exception:
                        pass

                checkpoint = torch.load(
                    model_info['path'],
                    map_location=self.device,
                    weights_only=True  # safe path (no code execution)
                )
            except TypeError:
                # Older torch without weights_only kwarg
                checkpoint = torch.load(model_info['path'], map_location=self.device)
            except Exception:
                # Fallback: allow full load if file is trusted
                checkpoint = torch.load(
                    model_info['path'],
                    map_location=self.device,
                    weights_only=False  # legacy behavior; may execute code
                )

            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.current_model = model
            self.current_model_info = model_info
            
            return f"✅ Model loaded successfully!\n\n**{model_key}**\n\n{model_info['description']}"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def predict_image(self, image):
        """Predict if image is real or fake"""
        if self.current_model is None:
            return "Please load a model first!", None
        
        if image is None:
            return "Please upload an image!", None
        
        try:
            # Preprocess image
            if isinstance(image, str):  # file path
                image = Image.open(image).convert('RGB')
            elif hasattr(image, 'convert'):  # PIL Image
                image = image.convert('RGB')
            
            # Transform image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.current_model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                fake_prob = probabilities[0][1].item()  # Probability of fake
                real_prob = probabilities[0][0].item()  # Probability of real
            
            # Create result text
            result_text = f"""
## 🔍 Detection Results

**Real Image Probability:** {real_prob*100:.2f}%
**Generated/Fake Image Probability:** {fake_prob*100:.2f}%

### 🎯 Prediction: {"**REAL IMAGE**" if real_prob > fake_prob else "**GENERATED/FAKE IMAGE**"}

---
**Model Used:** {list(self.available_models.keys())[list(self.available_models.values()).index(self.current_model_info)]}

**Confidence:** {max(real_prob, fake_prob)*100:.2f}%
            """
            
            # Create confidence chart data
            chart_data = {
                "Real": real_prob * 100,
                "Generated/Fake": fake_prob * 100
            }
            
            return result_text, chart_data
            
        except Exception as e:
            return f"❌ Error processing image: {str(e)}", None

def create_interface():
    """Create Gradio interface"""
    detector = T_GD_Interface()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .model-info {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-positive {
        color: #28a745;
        font-weight: bold;
    }
    .result-negative {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=css, title="T-GD: GAN-Generated Image Detection") as interface:
        gr.Markdown("""
        # 🔍 T-GD: Transferable GAN-Generated Image Detection
        
        Upload an image to detect if it's real or generated by a GAN (StyleGAN, StarGAN, PGGAN, etc.)
        
        **Steps:**
        1. Select a model from the dropdown
        2. Click "Load Model" 
        3. Upload an image
        4. Click "Detect Image"
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🤖 Model Selection")
                
                _available_keys = list(detector.available_models.keys())
                _default_key = _available_keys[0] if _available_keys else None
                model_dropdown = gr.Dropdown(
                    choices=_available_keys,
                    value=_default_key,
                    label="Available Models"
                )
                
                load_button = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Markdown("No model loaded")
                
                gr.Markdown("## 📁 Image Upload")
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image to Test"
                )
                
                detect_button = gr.Button("🔍 Detect Image", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Detection Results")
                
                result_text = gr.Markdown("Upload an image and click detect to see results...")
                
                with gr.Row():
                    confidence_chart = gr.Plot(label="Detection Confidence")
        
        # Model information section
        with gr.Accordion("📖 Model Information", open=False):
            gr.Markdown("""
            ### Available Models:
            
            **Pre-trained Models:**
            - Trained on specific GAN datasets (StarGAN, StyleGAN, etc.)
            - Good for detecting images from the same GAN type they were trained on
            
            **Transfer Learning Models (T-GD):**
            - Trained to transfer knowledge between different GAN types
            - Better generalization across different GAN architectures
            - Format: SOURCE → TARGET (e.g., StarGAN → StyleGAN2)
            
            ### Supported GAN Types:
            - **StarGAN**: Facial attribute transfer
            - **StyleGAN/StyleGAN2**: High-quality face generation  
            - **PGGAN**: Progressive growing GANs
            
            ### Architecture Options:
            - **EfficientNet**: Lightweight and efficient
            - **ResNeXt**: More robust feature extraction
            """)
        
        # Event handlers
        # 1) Keep manual load button
        load_button.click(
            fn=detector.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        # 2) Auto-load when dropdown selection changes
        model_dropdown.change(
            fn=detector.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        # 3) Auto-load default model on initial page load
        def _auto_load_default():
            try:
                keys = list(detector.available_models.keys())
                if not keys:
                    return "No models found in weights directory."
                return detector.load_model(keys[0])
            except Exception as e:
                return f"❌ Error during auto-load: {str(e)}"

        interface.load(fn=_auto_load_default, inputs=[], outputs=[model_status])
        
        def predict_and_chart(image):
            result_text, chart_data = detector.predict_image(image)
            if chart_data:
                # Create a simple bar chart using matplotlib
                import matplotlib.pyplot as plt
                
                categories = list(chart_data.keys())
                values = list(chart_data.values())
                colors = ['#28a745' if cat == 'Real' else '#dc3545' for cat in categories]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(categories, values, color=colors, alpha=0.7)
                
                ax.set_ylabel('Probability (%)')
                ax.set_title('Detection Confidence')
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                return result_text, fig
            return result_text, None
        
        detect_button.click(
            fn=predict_and_chart,
            inputs=[image_input],
            outputs=[result_text, confidence_chart]
        )
        
        # Example images section
        with gr.Accordion("🖼️ Example Usage", open=False):
            gr.Markdown("""
            ### Tips for Best Results:
            
            1. **Image Quality**: Use clear, high-resolution images
            2. **Model Selection**: 
               - Use pre-trained models for images from known GAN types
               - Use T-GD models for better cross-GAN detection
            3. **Confidence Interpretation**:
               - >90%: Very confident prediction
               - 70-90%: Good confidence
               - 50-70%: Uncertain, manual review recommended
               - <50%: Low confidence
            
            ### Model Performance:
            - Models are trained on 128x128 or 256x256 images
            - Performance may vary on images of different resolutions
            - Transfer learning models generally perform better across different GAN types
            """)
    
    return interface

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import gradio
    except ImportError:
        print("Installing Gradio...")
        os.system("pip install gradio")
        import gradio as gr
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
