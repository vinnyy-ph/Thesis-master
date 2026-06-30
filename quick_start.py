import os
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from models import EfficientNet
from options.test import TestOptions
from utils import AverageMeter, accuracy
from utils.reproducibility import set_seeds
from utils.metrics import compute_metrics
from utils import run_logger
from tqdm import tqdm
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



opt = TestOptions().parse(print_options=False)
print("{} from {} model testing on {}".format(opt.arch, opt.source_dataset, opt.target_dataset))

set_seeds(opt.seed)
run_dir = run_logger.start_run(opt)

gpu_id = opt.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
use_cuda = torch.cuda.is_available()
print("GPU device %d:" %(gpu_id), use_cuda)

model = EfficientNet.from_name(opt.arch, num_classes=opt.classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.resume:
    pretrained = opt.resume
    print("=> using pre-trained model '{}'".format(pretrained))
    model.load_state_dict(torch.load(pretrained, map_location=device)['state_dict'])

model.to(device)
if device.type == 'cuda':
    cudnn.benchmark = True
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

def test(val_loader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    y_true_all, y_score_all = [], []
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets.data)
            losses.update(loss.data.tolist(), inputs.size(0))
            probs = F.softmax(outputs, dim=1)[:, 1]
            y_true_all.append(targets.detach().cpu())
            y_score_all.append(probs.detach().cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    metrics = compute_metrics(torch.cat(y_true_all).numpy(),
                              torch.cat(y_score_all).numpy())
    print("acc:{accuracy:.4f} prec:{precision:.4f} rec:{recall:.4f} "
          "f1:{f1:.4f} ap:{average_precision} auroc:{auroc}".format(**metrics))
    return (losses.avg, metrics)

test_aug = transforms.Compose([
    transforms.Resize(opt.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data_dir = opt.source_dataset
test_dir = os.path.join(data_dir, 'test')
test_loader = DataLoader(datasets.ImageFolder(test_dir, test_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

print("Performance of {}".format(data_dir))
test_loss, test_metrics = test(test_loader, model, criterion, 1, use_cuda)

data_dir = opt.target_dataset
test_dir = os.path.join(data_dir, 'test')
test_loader = DataLoader(datasets.ImageFolder(test_dir, test_aug),
                       batch_size=opt.test_batch, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

print("Performance of {}".format(data_dir))
test_loss, test_metrics = test(test_loader, model, criterion, 1, use_cuda)

run_logger.finalize(run_dir, test_metrics)
