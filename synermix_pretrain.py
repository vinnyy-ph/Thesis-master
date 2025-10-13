import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models import EfficientNet
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from warmup_scheduler import GradualWarmupScheduler
from utils.aug import data_augment, rand_bbox
from utils.train_utils import save_checkpoint, adjust_learning_rate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from options.base import BaseOptions

opt = BaseOptions().parse(print_options=False)

# Add new SynerMix parameters to options
if not hasattr(opt, 'synermix_beta'):
    opt.synermix_beta = 0.5  # Balance between intra and inter class mixing (default 0.5)
if not hasattr(opt, 'synermix_alpha'):
    opt.synermix_alpha = 1.0  # Beta distribution parameter for inter-class mixing
if not hasattr(opt, 'use_synermix'):
    opt.use_synermix = True  # Enable SynerMix

gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)

# Modified model class to extract features before final classification
class SynerMixEfficientNet(nn.Module):
    def __init__(self, base_model):
        super(SynerMixEfficientNet, self).__init__()
        self.base_model = base_model
        # Instead of using ModuleList directly, use base_model's extract_features method
        self.classifier = base_model._fc
        
    def forward(self, x, return_features=False):
        # Use the base model's extract_features method
        features = self.base_model.extract_features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        
        if return_features:
            return features
        
        output = self.classifier(features)
        return output
    
    def forward_with_features(self, x):
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output, features

base_model = EfficientNet.from_name(opt.arch, num_classes=opt.classes,
                                   override_params={'dropout_rate': opt.dropout, 'drop_connect_rate': opt.dropconnect})
model = SynerMixEfficientNet(base_model)
model.to('cuda')
cudnn.benchmark = True

best_acc = 0
data_dir = opt.source_dataset
train_dir = os.path.join(data_dir, 'train')

train_aug = transforms.Compose([
    transforms.Lambda(lambda img: data_augment(img, opt)),
    transforms.Resize(opt.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(datasets.ImageFolder(train_dir, train_aug),
                         batch_size=opt.train_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

val_dir = os.path.join(data_dir, 'val')
val_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_loader = DataLoader(datasets.ImageFolder(val_dir, val_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

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

def supplement_batch(inputs, targets):
    """Supplement mini-batch to ensure each class has at least 2 samples"""
    unique_classes, class_counts = torch.unique(targets, return_counts=True)
    
    supplement_inputs = []
    supplement_targets = []
    
    for cls, count in zip(unique_classes, class_counts):
        if count == 1:
            # Find an additional sample from the same class
            cls_mask = targets == cls
            cls_indices = torch.where(cls_mask)[0]
            
            # Randomly duplicate the single sample (in practice, you'd sample from dataset)
            # For simplicity, we duplicate the existing sample with slight noise
            original_sample = inputs[cls_indices[0]].unsqueeze(0)
            # Add small noise to create variation
            noise = torch.randn_like(original_sample) * 0.01
            supplemented_sample = original_sample + noise
            
            supplement_inputs.append(supplemented_sample)
            supplement_targets.append(cls.unsqueeze(0))
    
    if supplement_inputs:
        supplement_inputs = torch.cat(supplement_inputs, dim=0)
        supplement_targets = torch.cat(supplement_targets, dim=0)
        
        inputs = torch.cat([inputs, supplement_inputs], dim=0)
        targets = torch.cat([targets, supplement_targets], dim=0)
    
    return inputs, targets

def intra_class_mixup(features, targets):
    """Implement intra-class feature mixing following SynerMix algorithm"""
    unique_classes = torch.unique(targets)
    batch_size = features.size(0)
    feature_dim = features.size(1)
    
    # Create synthesized features for each class
    synthesized_features = []
    synthesized_targets = []
    
    for cls in unique_classes:
        cls_mask = targets == cls
        cls_features = features[cls_mask]
        
        if cls_features.size(0) >= 2:  # Ensure at least 2 samples for mixing
            # Generate random weights for linear interpolation
            num_samples = cls_features.size(0)
            weights = torch.rand(num_samples).cuda()
            weights = weights / weights.sum()  # Normalize weights
            
            # Create synthesized feature representation
            synthesized_feature = torch.sum(weights.unsqueeze(1) * cls_features, dim=0, keepdim=True)
            synthesized_features.append(synthesized_feature)
            synthesized_targets.append(cls.unsqueeze(0))
    
    if synthesized_features:
        synthesized_features = torch.cat(synthesized_features, dim=0)
        synthesized_targets = torch.cat(synthesized_targets, dim=0)
        return synthesized_features, synthesized_targets
    else:
        return None, None

def inter_class_mixup(inputs, targets, alpha=1.0):
    """Implement traditional MixUp for inter-class mixing"""
    batch_size = inputs.size(0)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    # Random permutation for mixing
    rand_index = torch.randperm(batch_size).cuda()
    
    # Linear interpolation of inputs
    mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index]
    
    # Mixed targets (soft labels)
    targets_a = targets
    targets_b = targets[rand_index]
    
    return mixed_inputs, targets_a, targets_b, lam

def synermix_loss(outputs, targets_a, targets_b=None, lam=None):
    """Calculate loss for mixed samples"""
    if targets_b is not None:
        # Inter-class mixup loss
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    else:
        # Standard loss or intra-class loss
        return criterion(outputs, targets_a)

def train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()
    
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        if batch_size < opt.train_batch:
            continue
            
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        total_loss = 0.0
        
        if opt.use_synermix and opt.synermix_beta > 0:
            # SynerMix Implementation
            
            # Step 1: Supplement batch to ensure each class has at least 2 samples
            inputs, targets = supplement_batch(inputs, targets)
            
            # Step 2: Extract features from unaugmented images
            with torch.no_grad():
                features = model(inputs, return_features=True)
            
            # Step 3: Intra-class mixup component
            intra_features, intra_targets = intra_class_mixup(features, targets)
            
            if intra_features is not None:
                # Forward pass for intra-class synthesized features
                intra_outputs = model.classifier(intra_features)
                intra_loss = criterion(intra_outputs, intra_targets)
                
                # Calculate intra-class metrics
                intra_prec1 = accuracy(intra_outputs.data, intra_targets.data)
            else:
                intra_loss = torch.tensor(0.0).cuda()
                intra_prec1 = [0.0]
            
            # Step 4: Inter-class mixup component (traditional MixUp)
            mixed_inputs, targets_a, targets_b, lam = inter_class_mixup(inputs, targets, opt.synermix_alpha)
            inter_outputs = model(mixed_inputs)
            inter_loss = synermix_loss(inter_outputs, targets_a, targets_b, lam)
            
            # Calculate inter-class metrics
            inter_prec1 = accuracy(inter_outputs.data, targets_a.data)
            
            # Step 5: Combine losses with synergistic approach
            total_loss = opt.synermix_beta * intra_loss + (1 - opt.synermix_beta) * inter_loss
            
            # Use inter-class outputs for overall metrics (as they represent the full batch)
            outputs = inter_outputs
            main_targets = targets_a
            prec1 = inter_prec1
            
        else:
            # Original CutMix implementation (fallback)
            r = np.random.rand(1)
            if opt.cm_beta > 0 and r < opt.cm_prob:
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                tt = targets[rand_index]
                boolean = targets == tt
                rand_index = rand_index[boolean]
                lam = np.random.beta(opt.cm_beta, opt.cm_beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            outputs = model(inputs)
            total_loss = criterion(outputs, targets)
            main_targets = targets
            prec1 = accuracy(outputs.data, targets.data)
        
        # Calculate AUROC (assuming binary classification)
        if opt.classes == 2:
            try:
                auroc = roc_auc_score(main_targets.cpu().detach().numpy(), 
                                    torch.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1])
            except:
                auroc = 0.5  # Default value if AUROC calculation fails
        else:
            auroc = 0.0  # Multi-class AUROC not implemented here
        
        # Record metrics
        losses.update(total_loss.data.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        arc.update(auroc, inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 100 == 0:
            print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
                batch=batch_idx+1, size=len(train_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    
    print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
        batch=batch_idx+1, size=len(train_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    
    return (losses.avg, top1.avg, arc.avg)

def test(opt, val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    arc = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data)
            
            if opt.classes == 2:
                try:
                    auroc = roc_auc_score(targets.cpu().detach().numpy(), 
                                        torch.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1])
                except:
                    auroc = 0.5
            else:
                auroc = 0.0
            
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            arc.update(auroc, inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
            batch=batch_idx+1, size=len(val_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))
    
    return (losses.avg, top1.avg, arc.avg)

# Training loop
for epoch in range(start_epoch, opt.epochs):
    opt.lr = optimizer.state_dict()['param_groups'][0]['lr']
    adjust_learning_rate(optimizer, epoch, opt)
    
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))
    
    train_loss, train_acc, train_auroc = train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc, test_auroc = test(opt, val_loader, model, criterion, epoch, use_cuda)
    
    logger.append([opt.lr, train_loss, test_loss, train_acc, test_acc, train_auroc, test_auroc])
    
    scheduler_warmup.step()
    
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=opt.checkpoint)