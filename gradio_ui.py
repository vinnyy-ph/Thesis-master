import os
import sys
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
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
            
    def analyze_with_all_models(self, image, progress=gr.Progress()):
        """Run analysis with all available models sequentially"""
        if image is None:
            return "Please upload an image!", None, None
            
        try:
            # Preprocess image once
            if isinstance(image, str):  # file path
                image = Image.open(image).convert('RGB')
            elif hasattr(image, 'convert'):  # PIL Image
                image = image.convert('RGB')
                
            # Transform image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Store results
            results = {}
            combined_result = {"Real": 0.0, "Generated/Fake": 0.0}
            model_count = len(self.available_models)
            
            # Create a bar plot figure for overall results
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Track best models
            best_real_model = {"name": "", "prob": 0}
            best_fake_model = {"name": "", "prob": 0}
            
            # Process with each model
            progress(0, desc="Initializing...")
            for i, (model_key, model_info) in enumerate(self.available_models.items()):
                progress((i / model_count) * 100, desc=f"Processing with {model_key}...")
                
                try:
                    # Create and load model
                    if model_info['arch'] == 'efficientnet':
                        model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
                    else:  # resnext
                        model = resnext50_32x4d(num_classes=2)
                        
                    checkpoint = torch.load(model_info['path'], map_location=self.device)
                    
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.to(self.device)
                    model.eval()
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        
                        fake_prob = probabilities[0][1].item()  # Probability of fake
                        real_prob = probabilities[0][0].item()  # Probability of real
                    
                    # Store results for this model
                    results[model_key] = {
                        "Real": real_prob * 100,
                        "Generated/Fake": fake_prob * 100,
                        "Prediction": "REAL" if real_prob > fake_prob else "FAKE",
                        "Confidence": max(real_prob, fake_prob) * 100
                    }
                    
                    # Add to combined result (simple averaging ensemble)
                    combined_result["Real"] += real_prob * 100
                    combined_result["Generated/Fake"] += fake_prob * 100
                    
                    # Track best models
                    if real_prob > best_real_model["prob"]:
                        best_real_model = {"name": model_key, "prob": real_prob}
                    
                    if fake_prob > best_fake_model["prob"]:
                        best_fake_model = {"name": model_key, "prob": fake_prob}
                    
                    # Free memory
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    results[model_key] = {
                        "Error": str(e)
                    }
            
            # Average the combined results
            combined_result["Real"] /= model_count
            combined_result["Generated/Fake"] /= model_count
            
            # Generate detailed result text
            result_text = f"""
## 🔍 Ensemble Detection Results

**Real Image Probability:** {combined_result['Real']:.2f}%
**Generated/Fake Image Probability:** {combined_result['Generated/Fake']:.2f}%

### 🎯 Ensemble Prediction: {"**REAL IMAGE**" if combined_result['Real'] > combined_result['Generated/Fake'] else "**GENERATED/FAKE IMAGE**"}

---
### Best Models:
- Best model for detecting as real: **{best_real_model['name']}** ({best_real_model['prob']*100:.2f}%)
- Best model for detecting as fake: **{best_fake_model['name']}** ({best_fake_model['prob']*100:.2f}%)

### Model Breakdown:
"""
            
            # Sort models by confidence
            sorted_results = sorted(
                [(k, v) for k, v in results.items() if "Error" not in v],
                key=lambda x: x[1]["Confidence"],
                reverse=True
            )
            
            # Add top 3 most confident models to the result text
            for i, (model_key, result) in enumerate(sorted_results[:3]):
                result_text += f"""
#### {i+1}. {model_key}
- Prediction: **{result['Prediction']}**
- Confidence: {result['Confidence']:.2f}%
- Real: {result['Real']:.2f}% | Fake: {result['Generated/Fake']:.2f}%
"""
            
            # Create detailed results table visualization
            model_names = []
            real_probs = []
            fake_probs = []
            
            for model_key, result in sorted_results:
                if "Error" not in result:
                    model_names.append(model_key.split(' - ')[1] if ' - ' in model_key else model_key)
                    real_probs.append(result["Real"])
                    fake_probs.append(result["Generated/Fake"])
            
            # Create bar chart
            x = range(len(model_names))
            bar_width = 0.35
            
            ax.bar([i - bar_width/2 for i in x], real_probs, bar_width, label='Real', color='#28a745', alpha=0.7)
            ax.bar([i + bar_width/2 for i in x], fake_probs, bar_width, label='Fake', color='#dc3545', alpha=0.7)
            
            ax.set_ylabel('Probability (%)')
            ax.set_title('Detection Results Across All Models')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            
            # Add horizontal line at 50%
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.6)
            
            # Add overall decision
            overall_decision = "REAL" if combined_result['Real'] > combined_result['Generated/Fake'] else "FAKE"
            ax.text(
                0.5, 0.98, 
                f"ENSEMBLE PREDICTION: {overall_decision} IMAGE",
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='wheat', alpha=0.5)
            )
            
            plt.tight_layout()
            
            # Create single model comparison chart for real vs fake across models
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            models_sorted_by_diff = sorted(
                [(k, abs(v["Real"] - v["Generated/Fake"])) for k, v in results.items() if "Error" not in v],
                key=lambda x: x[1],
                reverse=True
            )[:8]  # Top 8 most confident models
            
            model_names_diff = []
            real_vals = []
            fake_vals = []
            
            for model_key, _ in models_sorted_by_diff:
                model_names_diff.append(model_key.split(' - ')[1] if ' - ' in model_key else model_key)
                real_vals.append(results[model_key]["Real"])
                fake_vals.append(results[model_key]["Generated/Fake"])
            
            # Create horizontal bar chart for the most confident models
            y_pos = range(len(model_names_diff))
            ax2.barh(y_pos, real_vals, 0.4, label='Real', color='#28a745', alpha=0.7)
            ax2.barh([y + 0.4 for y in y_pos], fake_vals, 0.4, label='Fake', color='#dc3545', alpha=0.7)
            
            ax2.set_yticks([y + 0.2 for y in y_pos])
            ax2.set_yticklabels(model_names_diff)
            ax2.set_xlabel('Probability (%)')
            ax2.set_title('Most Decisive Models')
            ax2.legend()
            
            plt.tight_layout()
            
            progress(100, desc="Analysis complete!")
            
            return result_text, fig, fig2
            
        except Exception as e:
            return f"❌ Error during multi-model analysis: {str(e)}", None, None

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
        """)
        
        with gr.Tabs():
            with gr.TabItem("Single Model Analysis"):
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
                        
                        detect_button = gr.Button("🔍 Detect with Selected Model", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("## 📊 Detection Results")
                        
                        result_text = gr.Markdown("Upload an image and click detect to see results...")
                        
                        with gr.Row():
                            confidence_chart = gr.Plot(label="Detection Confidence")
            
            with gr.TabItem("Ensemble Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 🔄 Analyze with All Models")
                        gr.Markdown("""
                        This will run your image through all available models and provide a comprehensive analysis.
                        
                        **Note:** This process may take some time as each model is loaded and run sequentially.
                        """)
                        
                        ensemble_image_input = gr.Image(
                            type="pil",
                            label="Upload Image for Ensemble Analysis"
                        )
                        
                        ensemble_button = gr.Button("🔍 Run Ensemble Analysis", variant="primary")
                        
                    with gr.Column(scale=2):
                        ensemble_result_text = gr.Markdown("Upload an image and click analyze to see combined results...")
                        
                        gr.Markdown("## 📊 Model Comparison")
                        ensemble_chart = gr.Plot(label="All Models Comparison")
                        
                        gr.Markdown("## 🎯 Most Decisive Models")
                        decisive_models_chart = gr.Plot(label="Most Decisive Models")
        
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
        
        # Event handlers for single model tab
        load_button.click(
            fn=detector.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        model_dropdown.change(
            fn=detector.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

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
        
        # Event handlers for ensemble tab
        ensemble_button.click(
            fn=detector.analyze_with_all_models,
            inputs=[ensemble_image_input],
            outputs=[ensemble_result_text, ensemble_chart, decisive_models_chart]
        )
    
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