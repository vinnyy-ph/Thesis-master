import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
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
from utils import Bar,Logger, AverageMeter, accuracy, mkdir_p, savefig
from warmup_scheduler import GradualWarmupScheduler
from utils.aug import data_augment, rand_bbox
from utils.train_utils import save_checkpoint, adjust_learning_rate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from options.base import BaseOptions

# ============== EARLY STOPPING CLASS (NEW) ==============
class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=True, delta=0, mode='max', save_path='checkpoint.pth.tar'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each validation improvement. 
                            Default: True
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
                            Default: 0
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the metric 
                       stops decreasing; in 'max' mode it will stop when the metric stops increasing.
                            Default: 'max'
            save_path (str): Path to save the best model checkpoint.
                            Default: 'checkpoint.pth.tar'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.save_path = save_path

        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - delta)
            self.best_score = float('inf')
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > (best + delta)
            self.best_score = float('-inf')

    def __call__(self, metric, model, epoch, optimizer, checkpoint_dir):
        """
        Call this at the end of each epoch with the validation metric.

        Args:
            metric (float): The validation metric to monitor (e.g., accuracy or loss)
            model: The PyTorch model
            epoch (int): Current epoch number
            optimizer: The optimizer
            checkpoint_dir (str): Directory to save checkpoints

        Returns:
            bool: True if training should stop, False otherwise
        """
        score = metric

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, checkpoint_dir, score)
        elif not self.monitor_op(score, self.best_score):
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            if self.verbose:
                improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
                print(f'Validation metric improved ({self.best_score:.6f} --> {score:.6f}). Saving model...')
            self.best_score = score
            self.save_checkpoint(model, epoch, optimizer, checkpoint_dir, score)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model, epoch, optimizer, checkpoint_dir, score):
        """Save model checkpoint when validation metric improves."""
        filepath = os.path.join(checkpoint_dir, 'early_stop_checkpoint.pth.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_score': self.best_score,
            'optimizer': optimizer.state_dict(),
        }, filepath)
# ========================================================

opt = BaseOptions().parse(print_options=False)
#print("{} from {} model testing on {}".format(opt.arch, opt.source_dataset, opt.target_dataset))
gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)

model = EfficientNet.from_name(opt.arch, num_classes=opt.classes,
                                 override_params={'dropout_rate': opt.dropout, 'drop_connect_rate': opt.dropconnect})
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
    # transforms.RandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0, inplace=True),
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

# ============== INITIALIZE EARLY STOPPING (NEW) ==============
# You can adjust these parameters:
# - patience: number of epochs to wait before stopping (default: 7)
# - delta: minimum change to qualify as improvement (default: 0)
# - mode: 'max' for accuracy, 'min' for loss (default: 'max')
early_stopping = EarlyStopping(
    patience=10,           # Wait 10 epochs without improvement
    verbose=True,          # Print messages
    delta=0.001,          # Minimum improvement of 0.1%
    mode='max'            # Monitor validation accuracy (use 'min' for loss)
)
# =============================================================

# Resume
if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = os.path.dirname(opt.resume)
    # checkpoint = torch.load(resume)
    resume = torch.load(opt.resume)
    best_acc = resume['best_acc']
    start_epoch = resume['epoch']
    model.load_state_dict(resume['state_dict'])
    optimizer.load_state_dict(resume['optimizer'])
    logger = Logger(os.path.join(checkpoint, 'log.txt'), resume=True)
else:
    logger = Logger(os.path.join(opt.checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train AUROC.', 'Valid AUROC.'])

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

        r = np.random.rand(1)
        if opt.cm_beta > 0 and r < opt.cm_prob:
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            tt= targets[rand_index]
            boolean = targets==tt
            rand_index = rand_index[boolean]
            lam = np.random.beta(opt.cm_beta, opt.cm_beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[boolean, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
        losses.update(loss.data.tolist(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        arc.update(auroc, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
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
            auroc = roc_auc_score(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()[:,1])
            losses.update(loss.data.tolist(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            arc.update(auroc, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('{batch}/{size} | Loss:{loss:.4f} | top1:{tp1:.4f} | AUROC:{ac:.4f}'.format(
        batch=batch_idx+1, size=len(val_loader), loss=losses.avg, tp1=top1.avg, ac=arc.avg))

    return (losses.avg, top1.avg, arc.avg)

# ============== MODIFIED TRAINING LOOP WITH EARLY STOPPING ==============
for epoch in range(opt.start_epoch, opt.epochs):
    opt.lr = optimizer.state_dict()['param_groups'][0]['lr']
    adjust_learning_rate(optimizer, epoch, opt)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))

    train_loss, train_acc, train_auroc = train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda)
    test_loss, test_acc, test_auroc = test(opt, val_loader, model, criterion, epoch, use_cuda)

    logger.append([opt.lr, train_loss, test_loss, train_acc, test_acc, train_auroc, test_auroc])
    scheduler_warmup.step()

    # Original checkpoint saving (still included)
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict' : model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=opt.checkpoint)

    # ============== EARLY STOPPING CHECK (NEW) ==============
    # Monitor validation accuracy (or use test_loss if you prefer to monitor loss)
    if early_stopping(test_acc, model, epoch + 1, optimizer, opt.checkpoint):
        print(f'\n=== Early stopping triggered after {epoch + 1} epochs ===')
        print(f'Best validation accuracy: {early_stopping.best_score:.4f}')
        print(f'Best model saved at: {opt.checkpoint}/early_stop_checkpoint.pth.tar')
        break
    # ========================================================

print('\n=== Training Complete ===')
print(f'Best validation accuracy: {best_acc:.4f}')