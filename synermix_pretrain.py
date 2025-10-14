import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import random

from models import EfficientNet
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from warmup_scheduler import GradualWarmupScheduler
from utils.aug import data_augment, rand_bbox
from utils.train_utils import save_checkpoint, adjust_learning_rate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from options.base import BaseOptions

opt = BaseOptions().parse(print_options=False)

# Set default dataset if not provided
if not opt.source_dataset:
    opt.source_dataset = './dataset/StyleGAN2_256'  # Use StyleGAN2_256 as default

# SynerMix specific parameters
if not hasattr(opt, 'synermix_beta'):
    opt.synermix_beta = 0.5  # Balance between intra and inter class mixing
if not hasattr(opt, 'synermix_alpha'):
    opt.synermix_alpha = 1.0  # Beta distribution parameter for inter-class mixing
if not hasattr(opt, 'synermix_warmup_epochs'):
    opt.synermix_warmup_epochs = 5  # Warmup period before applying SynerMix
if not hasattr(opt, 'synermix_min_samples'):
    opt.synermix_min_samples = 2  # Minimum samples per class for intra-class mixing

print(f"Using SynerMix with beta={opt.synermix_beta}, alpha={opt.synermix_alpha}, warmup={opt.synermix_warmup_epochs}")

# GPU setup
gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" % (gpu_id), use_cuda)

# Model with feature extraction capability
class SynerMixEfficientNet(nn.Module):
    def __init__(self, base_model):
        super(SynerMixEfficientNet, self).__init__()
        self.base_model = base_model

        # Keep references to components for feature processing
        self._avg_pooling = base_model._avg_pooling
        self._dropout = base_model._dropout
        self._fc = base_model._fc

    def extract_features(self, x):
        """Extract features using the built-in extract_features method"""
        return self.base_model.extract_features(x)

    def forward(self, x, return_features=False):
        # Extract features using the built-in method
        features = self.extract_features(x)

        if return_features:
            return features

        # Apply pooling and flattening (same as original forward method)
        features = self._avg_pooling(features)
        bs = x.size(0)
        features = features.view(bs, -1)

        if self.training:
            features = self._dropout(features)

        return self._fc(features)

# Create base model and wrap it for feature extraction
base_model = EfficientNet.from_name(opt.arch, num_classes=opt.classes,
                                   override_params={'dropout_rate': opt.dropout, 'drop_connect_rate': opt.dropconnect})
model = SynerMixEfficientNet(base_model)
model.to('cuda')
cudnn.benchmark = True
best_acc = 0

# Data loading
data_dir = opt.source_dataset
train_dir = os.path.join(data_dir, 'train')
train_aug = transforms.Compose([
    transforms.Lambda(lambda img: data_augment(img, opt)),
    transforms.Resize(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.ImageFolder(train_dir, train_aug)
train_loader = DataLoader(train_dataset,
                         batch_size=opt.train_batch, shuffle=True, 
                         num_workers=opt.num_workers, pin_memory=True)

val_dir = os.path.join(data_dir, 'val')
val_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_loader = DataLoader(datasets.ImageFolder(val_dir, val_aug),
                       batch_size=opt.test_batch, shuffle=True, 
                       num_workers=opt.num_workers, pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-4)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

# Resume functionality
if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = os.path.dirname(opt.resume)
    resume = torch.load(opt.resume)
    best_acc = resume['best_acc']
    start_epoch = resume['epoch']
    model.load_state_dict(resume['state_dict'])
    optimizer.load_state_dict(resume['optimizer'])
    logger = Logger(os.path.join(checkpoint, 'log.txt'), resume=True)
else:
    logger = Logger(os.path.join(opt.checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train AUROC.', 'Valid AUROC.'])

def adjust_synermix_params(epoch, total_epochs):
    """
    Dynamically adjust SynerMix parameters during training
    This implements the warmup and parameter scheduling strategy
    """
    # During warmup, don't use SynerMix (use standard training)
    if epoch < opt.synermix_warmup_epochs:
        return 0.0  # No mixing during warmup
    
    # Calculate training progress after warmup
    progress = (epoch - opt.synermix_warmup_epochs) / (total_epochs - opt.synermix_warmup_epochs)
    
    # Gradually shift from intra-class to inter-class mixing
    # This helps the model first learn good class representations (intra-class)
    # Then learn better decision boundaries (inter-class)
    if progress < 0.3:
        return 0.8  # Heavy emphasis on intra-class mixing in early epochs
    elif progress < 0.6:
        return 0.5  # Balanced mixing in middle epochs
    elif progress < 0.8:
        return 0.3  # More inter-class mixing in later epochs
    else:
        return 0.2  # Heavy emphasis on inter-class mixing in final epochs

def supplement_batch(inputs, targets, dataset, min_samples=2):
    """
    Supplement mini-batch to ensure each class has at least min_samples samples
    Implements algorithm lines 3-9 from the SynerMix paper
    """
    device = inputs.device
    batch_size = inputs.size(0)
    
    # Group samples by class
    unique_classes = torch.unique(targets)
    class_counts = {cls.item(): (targets == cls).sum().item() for cls in unique_classes}
    
    # Find classes with fewer than min_samples samples
    extra_inputs = []
    extra_targets = []
    
    for cls, count in class_counts.items():
        if count < min_samples:
            # Count how many extra samples we need
            extra_count = min_samples - count
            
            # Find all samples of this class in the dataset
            class_indices = []
            for idx, (_, label) in enumerate(dataset):
                if label == cls:
                    class_indices.append(idx)
            
            # If we don't have enough samples in the dataset, duplicate existing ones
            if len(class_indices) <= count:
                cls_idx = torch.where(targets == cls)[0]
                for i in range(extra_count):
                    idx = cls_idx[i % len(cls_idx)]
                    extra_inputs.append(inputs[idx].unsqueeze(0))
                    extra_targets.append(targets[idx].unsqueeze(0))
            else:
                # Sample new examples not in the current batch
                current_indices = torch.where(targets == cls)[0].cpu().numpy()
                available_indices = list(set(class_indices) - set(current_indices))
                
                if available_indices:
                    for i in range(extra_count):
                        idx = random.choice(available_indices)
                        img, label = dataset[idx]
                        extra_inputs.append(img.unsqueeze(0).to(device))
                        extra_targets.append(torch.tensor([label]).to(device))
    
    # If we have extra samples, add them to the batch
    if extra_inputs:
        extra_inputs = torch.cat(extra_inputs, dim=0)
        extra_targets = torch.cat(extra_targets, dim=0)
        inputs = torch.cat([inputs, extra_inputs], dim=0)
        targets = torch.cat([targets, extra_targets], dim=0)
    
    return inputs, targets

def intra_class_mixup(features_by_class):
    """
    Perform intra-class feature mixing
    Implements algorithm lines 10-16 from the SynerMix paper
    """
    mixed_features = []
    mixed_targets = []
    
    # For each class, perform feature mixing
    for cls, features in features_by_class.items():
        if features.size(0) >= 2:  # Need at least 2 samples for mixing
            # Sample random weights for each feature (line 12)
            num_samples = features.size(0)
            r = torch.rand(num_samples).to(features.device)
            
            # Normalize weights (line 13)
            weights = r / r.sum()
            
            # Create mixed feature as weighted sum (line 14)
            weights = weights.view(-1, 1)
            mixed_feature = (features * weights).sum(dim=0, keepdim=True)
            
            # Create one-hot target (line 15)
            mixed_targets.append(torch.tensor([cls]).to(features.device))
            mixed_features.append(mixed_feature)
    
    if mixed_features:
        try:
            # Concatenate all mixed features and targets
            return torch.cat(mixed_features, dim=0), torch.cat(mixed_targets, dim=0)
        except RuntimeError as e:
            print(f"Error concatenating mixed features: {e}")
            # Return empty tensors on error
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.tensor([]).to(device), torch.tensor([]).to(device)
    else:
        # Return empty tensors with proper device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor([]).to(device), torch.tensor([]).to(device)

def augment_images(inputs):
    """
    Apply augmentation to images (algorithm line 18)
    This function applies standard augmentations to the input images
    """
    # We'll use the existing data augmentation from the training pipeline
    # This is already handled by the transforms in the dataloader
    # But we could add additional augmentations here if needed
    return inputs

def inter_class_mixup(inputs, targets, alpha):
    """
    Perform inter-class image-level mixing 
    Implements algorithm lines 18-25 from the SynerMix paper
    """
    batch_size = inputs.size(0)
    
    # Apply augmentation to images (line 18)
    augmented_inputs = augment_images(inputs)
    
    # Shuffle inputs and targets (line 19)
    indices = torch.randperm(batch_size).to(inputs.device)
    shuffled_inputs = augmented_inputs[indices]
    shuffled_targets = targets[indices]
    
    mixed_inputs = []
    mixed_targets_a = []
    mixed_targets_b = []
    lambdas = []
    
    # For each sample in the batch (lines 20-25)
    for i in range(batch_size):
        # Sample lambda from Beta(alpha, alpha) (line 21)
        lam = np.random.beta(alpha, alpha)
        lambdas.append(lam)
        
        # Mix images using CutMix (line 22)
        # CutMix is more effective than simple mixup for image data
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs[i:i+1].size(), lam)
        
        mixed_input = inputs[i:i+1].clone()
        mixed_input[:, :, bbx1:bbx2, bby1:bby2] = shuffled_inputs[i:i+1, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on the area ratio
        actual_lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        
        mixed_inputs.append(mixed_input)
        mixed_targets_a.append(targets[i:i+1])
        mixed_targets_b.append(shuffled_targets[i:i+1])
    
    # Concatenate all mixed samples
    mixed_inputs = torch.cat(mixed_inputs, dim=0)
    mixed_targets_a = torch.cat(mixed_targets_a, dim=0)
    mixed_targets_b = torch.cat(mixed_targets_b, dim=0)
    
    return mixed_inputs, mixed_targets_a, mixed_targets_b, torch.tensor(lambdas).mean().item()

def train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()
    
    end = time.time()
    
    # Determine whether to use SynerMix based on warmup epoch
    use_synermix = epoch >= opt.synermix_warmup_epochs
    
    # Dynamically adjust synermix beta
    if use_synermix:
        opt.synermix_beta = adjust_synermix_params(epoch, opt.epochs)
        print(f"Epoch {epoch+1}: Using SynerMix with beta={opt.synermix_beta:.2f}")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            
        batch_size = inputs.size(0)

        if use_synermix:
            # Step 1: Supplement batch to ensure each class has at least min_samples samples (lines 3-9)
            inputs, targets = supplement_batch(inputs, targets, train_dataset, min_samples=opt.synermix_min_samples)
            
            # Get unique classes in the batch
            unique_classes = torch.unique(targets)
            
            # Step 2: Extract features for each class (lines 10-11)
            features_by_class = {}
            for cls in unique_classes:
                cls_idx = torch.where(targets == cls)[0]
                if len(cls_idx) >= 2:  # Only process classes with at least 2 samples
                    cls_inputs = inputs[cls_idx]
                    # Extract features for this class (gradients flow through)
                    features = model.extract_features(cls_inputs)
                    features_by_class[cls] = features
            
            # Step 3: Perform intra-class feature mixing (lines 12-16)
            if features_by_class:
                intra_features, intra_targets = intra_class_mixup(features_by_class)
            else:
                intra_features, intra_targets = None, None
            
            # Check if we have intra-class features
            if intra_features is not None and len(intra_features) > 0:
                # Apply pooling and flattening to intra-class features (same as main forward)
                intra_features_pooled = model._avg_pooling(intra_features)
                intra_features_flat = intra_features_pooled.view(intra_features.size(0), -1)

                if model.training:
                    intra_features_flat = model._dropout(intra_features_flat)

                # Pass mixed features through final classification layer
                intra_outputs = model._fc(intra_features_flat)

                # Calculate intra-class loss
                intra_loss = criterion(intra_outputs, intra_targets)

                # Get accuracy for intra-class mixing
                intra_prec1 = accuracy(intra_outputs.data, intra_targets.data)
            else:
                # No intra-class mixing possible
                intra_loss = torch.tensor(0.0).cuda()
                intra_prec1 = [0.0]
            
            # Step 4: Perform inter-class image mixing
            inter_inputs, inter_targets_a, inter_targets_b, lam = inter_class_mixup(inputs, targets, opt.synermix_alpha)
            
            # Forward pass for inter-class mixing
            inter_outputs = model(inter_inputs)
            
            # Calculate inter-class loss
            inter_loss = lam * criterion(inter_outputs, inter_targets_a) + (1 - lam) * criterion(inter_outputs, inter_targets_b)

            # Calculate total loss (line 27)
            # L_total = β · L_intra + (1 - β) · L_inter
            if intra_loss > 0:
                # We have both intra and inter losses
                loss = opt.synermix_beta * intra_loss + (1 - opt.synermix_beta) * inter_loss
            else:
                # Only inter-class loss (no intra-class mixing possible)
                loss = inter_loss

            # Measure accuracy using inter-class outputs (as they include all original samples)
            outputs = inter_outputs
            prec1 = accuracy(outputs.data, inter_targets_a.data)
            
        else:
            # Standard training or CutMix during warmup
            if hasattr(opt, 'cm_beta') and opt.cm_beta > 0 and np.random.rand() < getattr(opt, 'cm_prob', 0.5):
                # Apply CutMix
                rand_index = torch.randperm(batch_size).cuda()
                targets_b = targets[rand_index]
                lam = np.random.beta(opt.cm_beta, opt.cm_beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                
                # Adjust lambda
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                
                # Forward pass
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets_b)
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            prec1 = accuracy(outputs.data, targets.data)
        
        # Compute AUROC
        try:
            output_softmax = F.softmax(outputs, dim=1)
            if output_softmax.size(1) == 2:  # Binary classification
                auroc = roc_auc_score(targets.cpu().detach().numpy(), 
                                     output_softmax.cpu().detach().numpy()[:, 1])
            else:  # Multi-class - use one-vs-rest approach
                auroc = roc_auc_score(
                    torch.nn.functional.one_hot(targets, num_classes=output_softmax.size(1)).cpu().detach().numpy(), 
                    output_softmax.cpu().detach().numpy(), 
                    multi_class='ovr'
                )
        except ValueError:
            auroc = 0.5
            
        # Record metrics
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        arc.update(auroc, inputs.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if batch_idx % 100 == 0:
            mode = 'SynerMix' if use_synermix else 'Standard'
            print(f'[{mode}] {batch_idx}/{len(train_loader)} | Loss:{losses.avg:.4f} | Top1:{top1.avg:.4f} | AUROC:{arc.avg:.4f}')
    
    return (losses.avg, top1.avg, arc.avg)

def test(opt, val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()
    
    # Switch to evaluate mode
    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data)
            
            # Compute AUROC
            try:
                output_softmax = F.softmax(outputs, dim=1)
                if output_softmax.size(1) == 2:  # Binary classification
                    auroc = roc_auc_score(targets.cpu().detach().numpy(), 
                                         output_softmax.cpu().detach().numpy()[:, 1])
                else:  # Multi-class
                    auroc = roc_auc_score(
                        torch.nn.functional.one_hot(targets, num_classes=output_softmax.size(1)).cpu().detach().numpy(), 
                        output_softmax.cpu().detach().numpy(), 
                        multi_class='ovr'
                    )
            except ValueError:
                auroc = 0.5
                
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            arc.update(auroc, inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    print(f'Validation: Loss:{losses.avg:.4f} | Top1:{top1.avg:.4f} | AUROC:{arc.avg:.4f}')
    
    return (losses.avg, top1.avg, arc.avg)

# Training loop
for epoch in range(opt.start_epoch, opt.epochs):
    opt.lr = optimizer.state_dict()['param_groups'][0]['lr']
    
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))
    if epoch >= opt.synermix_warmup_epochs:
        print('SynerMix Beta: %.2f' % opt.synermix_beta)
    
    train_loss, train_acc, train_auroc = train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc, test_auroc = test(opt, val_loader, model, criterion, epoch, use_cuda)
    
    logger.append([opt.lr, train_loss, test_loss, train_acc, test_acc, train_auroc, test_auroc])
    
    # Step learning rate scheduler
    scheduler_warmup.step()
    
    # Save checkpoint
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=opt.checkpoint)
    
    print(f'Best accuracy: {best_acc:.2f}%')