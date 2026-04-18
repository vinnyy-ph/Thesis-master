import os
import torch
from torch import nn
import gradio as gr
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import glob
import time
from PIL import ImageFile
import numpy as np

# Adjust inputs based on actual path of models
try:
    from models import EfficientNet
    from resnext import resnext50_32x4d
    from utils import AverageMeter
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models import EfficientNet
    from resnext import resnext50_32x4d
    from utils import AverageMeter

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_available_models():
    """Scan weights directory for available models"""
    models = {}
    
    # Pre-trained models
    for dir_name in ['pre-train', 't-gd']:
        weight_dir = os.path.join("weights", dir_name)
        if os.path.exists(weight_dir):
            for arch in ['efficientnet', 'resnext']:
                arch_path = os.path.join(weight_dir, arch)
                if os.path.exists(arch_path):
                    for pattern in ["*.pth.tar", "*.pth", "*.pt"]:
                        for weight_file in glob.glob(os.path.join(arch_path, pattern)):
                            model_name = os.path.basename(weight_file)
                            for ext in ['.pth.tar', '.pth', '.pt']:
                                model_name = model_name.replace(ext, '')
                            
                            prefix = "Pre-trained" if dir_name == 'pre-train' else "T-GD"
                            model_key = f"{prefix}: {arch.upper()} - {model_name.upper()}"
                            
                            if model_key not in models:
                                models[model_key] = {
                                    'path': weight_file,
                                    'arch': arch,
                                    'type': dir_name
                                }
    return models

def load_model_weights(model_info, device):
    if model_info['arch'] == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
    else:
        model = resnext50_32x4d(num_classes=2)
        
    try:
        try:
            from numpy.core.multiarray import scalar as numpy_scalar
        except Exception:
            numpy_scalar = None

        if hasattr(torch, 'serialization') and numpy_scalar is not None:
            try:
                torch.serialization.add_safe_globals([numpy_scalar])
            except Exception:
                pass
                
        checkpoint = torch.load(model_info['path'], map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_info['path'], map_location=device)
    except Exception:
        checkpoint = torch.load(model_info['path'], map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    from_synermix = any('base_model.' in k for k in state_dict.keys())
    if from_synermix:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('base_model.'):
                new_key = k[len('base_model.'):]
                new_state_dict[new_key] = v
            elif k == 'classifier.weight':
                new_state_dict['_fc.weight'] = v
            elif k == 'classifier.bias':
                new_state_dict['_fc.bias'] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def run_evaluation(model_key, dataset_path, batch_size, num_workers, img_size, gpu_id, progress=gr.Progress()):
    if not model_key:
        return "❌ Error: No model selected."
        
    if not dataset_path or not os.path.exists(dataset_path):
        return f"❌ Error: Dataset path '{dataset_path}' does not exist."
        
    if os.path.isfile(dataset_path):
        dataset_path = os.path.dirname(dataset_path)
        
    test_dir = os.path.join(dataset_path, 'test')
    if not os.path.exists(test_dir):
        return f"❌ Error: 'test' directory not found in '{dataset_path}'. Expected structure is '{test_dir}'."
        
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(gpu_id))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = get_available_models()
    if model_key not in models:
        return "❌ Error: Model not found."
        
    progress(0, desc="Loading model...")
    try:
        model = load_model_weights(models[model_key], device)
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"
        
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    test_aug = transforms.Compose([
        transforms.Resize(int(img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    progress(0.1, desc="Loading dataset...")
    try:
        test_dataset = datasets.ImageFolder(test_dir, test_aug)
        test_loader = DataLoader(test_dataset, batch_size=int(batch_size), 
                               shuffle=False, num_workers=int(num_workers), pin_memory=True)
    except Exception as e:
        return f"❌ Error loading dataset: {str(e)}"

    progress(0.2, desc="Evaluating...")
    losses = AverageMeter()
    
    total_batches = len(test_loader)
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
                
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            losses.update(loss.item(), inputs.size(0))
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            progress(0.2 + 0.7 * (i / total_batches), desc=f"Evaluating batch {i+1}/{total_batches}")
            
    # Calculate final metrics
    progress(0.95, desc="Computing final metrics...")
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    try:
        auroc = roc_auc_score(all_targets, all_outputs[:, 1])
    except ValueError:
        auroc = 0.0 # Handle case where only one class is present
        
    preds = np.argmax(all_outputs, axis=1)
    acc = accuracy_score(all_targets, preds)
    
    result_html = f'''
    <div style="padding: 20px; background-color: #f8f9fa; border-radius: 8px; border-left: 5px solid #007bff;">
        <h3 style="margin-top:0; color:#007bff;">Evaluation Results</h3>
        <p><b>Dataset:</b> {dataset_path}</p>
        <p><b>Model:</b> {model_key}</p>
        <hr/>
        <ul style="font-size: 1.1em; line-height: 1.6;">
            <li><b>Loss:</b> {losses.avg:.4f}</li>
            <li><b>Accuracy:</b> {acc*100:.2f}%</li>
            <li><b>AUROC:</b> {auroc:.4f}</li>
            <li><b>Total Images:</b> {len(test_dataset)}</li>
        </ul>
    </div>
    '''
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return result_html

def create_interface():
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )
    
    models = get_available_models()
    model_choices = list(models.keys())
    
    with gr.Blocks(theme=theme, title="T-GD Model Evaluation") as app:
        gr.Markdown('''
        # 📊 T-GD Model Evaluation Interface
        Evaluate pre-trained and T-GD models on your datasets. 
        ''')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Select Model",
                    info="Choose a model from weights directory"
                )
                
                dataset_path = gr.FileExplorer(
                    label="Dataset Path",
                    root_dir=".",
                    file_count="single"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    batch_size = gr.Number(value=32, label="Batch Size", precision=0)
                    num_workers = gr.Number(value=4, label="Num Workers", precision=0)
                    img_size = gr.Number(value=128, label="Image Size", precision=0)
                    gpu_id = gr.Number(value=0, label="GPU ID", precision=0)
                    
                eval_btn = gr.Button("🚀 Run Evaluation", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Results")
                result_html = gr.HTML(
                    value='<div style="padding:20px; text-align:center; color:#666;">Evaluation results will appear here.</div>'
                )
                
        eval_btn.click(
            fn=run_evaluation,
            inputs=[model_dropdown, dataset_path, batch_size, num_workers, img_size, gpu_id],
            outputs=result_html
        )
        
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False)
