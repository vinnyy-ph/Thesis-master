# Phase 1: Fix the Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the T-GD + SynerMix codebase honest — every entry point runs end-to-end, the SynerMix training loop is reachable, and the duplicate EfficientNet package is gone.

**Architecture:** Surgical fixes to existing scripts; no new abstractions. A tiny synthetic ImageFolder dataset (`dataset/_smoke/`) lets every training/eval script smoke-run on CPU in seconds. The canonical model package is `models/` (GroupNorm, `_gn*` state-dict keys — verified to match the bundled weights, which contain 98 `_gn` keys and 0 `_bn` keys). The `EfficientNet/` package is a BatchNorm variant whose state-dict keys are **incompatible** with the bundled weights; it gets deleted. `resnext/` STAYS — `gradio_ui.py` instantiates `resnext50_32x4d` at lines 108, 264, 606.

**Tech Stack:** Python 3.12 (`.\venv\Scripts\python.exe`), PyTorch 2.12 (CPU), torchvision, PIL, numpy. Windows / PowerShell.

**Critical environment notes for every step:**
- ALWAYS prefix Python runs with `$env:PYTHONUTF8='1'; ` — several scripts print emoji/✓ characters that crash Windows' default cp1252 console encoding with `UnicodeEncodeError`.
- ALWAYS pass `--num_workers 0` to training/eval scripts — the train transforms use `transforms.Lambda`, which cannot be pickled by Windows DataLoader worker processes.
- Run everything from the repo root: `C:\Users\Vincent\Documents\GitHub\Portfolio\Thesis-master`.
- `dataset/` and `log/` are already in `.gitignore`; smoke artifacts will not pollute `git status`.
- Phases 2–4 of the spec get their own plans after this one lands.

---

### Task 1: Commit the pending device-portability work

The working tree has 9 modified files. Eight of them are a coherent, finished change (CUDA→device-agnostic handling, and an `os.mkdirs` crash fix in options). The ninth, `synermix_pretrain.py`, contains a bug fixed in Task 3 — do NOT commit it here.

**Files:**
- Commit (A): `early_stop_pretrain.py`, `eval.py`, `pretrain.py`, `quick_start.py`, `transfer.py`, `test_synermix.py`
- Commit (B): `options/base.py`, `options/transfer.py`

- [ ] **Step 1: Baseline sanity — run the existing test script**

Run:
```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: ends with `All tests passed successfully!` (exit code 0). Takes ~1–2 min on CPU (builds an EfficientNet-b0 twice). If this fails, STOP and report — the baseline is broken in a way this plan didn't anticipate.

- [ ] **Step 2: Commit the device-portability changes**

```powershell
git add early_stop_pretrain.py eval.py pretrain.py quick_start.py transfer.py test_synermix.py
git commit -m "fix: make training and eval scripts device-agnostic (CPU fallback)"
```

- [ ] **Step 3: Commit the options crash fix**

```powershell
git add options/base.py options/transfer.py
git commit -m "fix: replace non-existent os.mkdirs with guarded os.makedirs in options"
```

- [ ] **Step 4: Verify only synermix_pretrain.py remains dirty**

Run: `git status --short`
Expected: ` M synermix_pretrain.py` is the only modified tracked file.

---

### Task 2: Smoke-dataset fixture

A tiny synthetic dataset so every entry point can run on CPU in seconds. Real GAN data is irrelevant here — we are testing code paths, not accuracy.

**Files:**
- Create: `scripts/make_smoke_dataset.py`

- [ ] **Step 1: Write the fixture script**

Create `scripts/make_smoke_dataset.py`:

```python
"""Create a tiny synthetic ImageFolder dataset for CPU smoke-testing the
training/eval entry points. Layout:

dataset/_smoke/
    train/{fake,real}/             8 images each
    val/{fake,real}/               8 images each
    test/{fake,real}/              8 images each
    style2/2000_shot/{fake,real}/  8 images each  (transfer.py train dir)

Also pre-creates the log/smoke* checkpoint dirs the smoke runs write to.
Run from the repo root: python scripts/make_smoke_dataset.py
"""
import os

import numpy as np
from PIL import Image

ROOT = os.path.join('dataset', '_smoke')
SPLITS = ['train', 'val', 'test', os.path.join('style2', '2000_shot')]
CLASSES = ['fake', 'real']
N_IMAGES = 8
SIZE = 64

rng = np.random.default_rng(7)

for split in SPLITS:
    for cls in CLASSES:
        d = os.path.join(ROOT, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(N_IMAGES):
            arr = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(d, f'{cls}_{i}.png'))

for sub in ['smoke', 'smoke_transfer', 'smoke_pretrain', 'smoke_es']:
    os.makedirs(os.path.join('log', sub), exist_ok=True)

print(f'Smoke dataset written to {ROOT}')
```

- [ ] **Step 2: Run it**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe scripts\make_smoke_dataset.py
```
Expected: `Smoke dataset written to dataset\_smoke`, and `dataset/_smoke/train/fake/` contains 8 PNGs.

- [ ] **Step 3: Commit**

```powershell
git add scripts/make_smoke_dataset.py
git commit -m "test: add synthetic smoke-dataset generator for CPU runs"
```

---

### Task 3: Make the SynerMix training loop reachable

`synermix_pretrain.py`'s working-tree version wraps setup in `if __name__ == '__main__':` (good — makes the module importable by `test_synermix.py`), but the training loop at lines 570–597 was indented into the body of `def test(...)` AFTER its `return` statement. It is dead code: the script sets up the model and data, then exits without training a single epoch.

Restructure: setup + loop move into a `main()` function at the end of the file; `train()`/`test()` take `device` (and `train()` takes `train_dataset`) as parameters instead of relying on globals; debug prints flagged in the audit are removed.

**Files:**
- Modify: `synermix_pretrain.py`

- [ ] **Step 1: Demonstrate the bug**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe synermix_pretrain.py --source_dataset dataset/_smoke --epochs 2 --synermix_warmup_epochs 1 --train_batch 4 --test_batch 8 --num_workers 0 --size 64 --checkpoint ./log/smoke
```
Expected (the bug): script prints the setup lines (`Using SynerMix with beta=...`, `GPU device 0: False`) then exits WITHOUT any `Epoch: [1 | 2]` output and WITHOUT writing `log/smoke/checkpoint.pth.tar`. Verify:
```powershell
Test-Path log/smoke/checkpoint.pth.tar
```
Expected: `False`

- [ ] **Step 2: Remove the debug print in `SynerMixEfficientNet.extract_features`**

In `synermix_pretrain.py`, replace:

```python
        # Store feature dimension for debugging
        if self.feature_dim is None:
            self.feature_dim = features.size(1)
            print(f"Feature dimension: {self.feature_dim}")
```

with:

```python
        if self.feature_dim is None:
            self.feature_dim = features.size(1)
```

- [ ] **Step 3: Replace the `if __name__ == '__main__':` setup block with nothing (it moves into `main()`)**

Delete the entire current `if __name__ == '__main__':` block (the one starting with `# Create base model and wrap it for feature extraction` and ending with `logger.set_names([...])`). The function definitions (`adjust_synermix_params`, `supplement_batch`, `intra_class_mixup`, `augment_images`, `inter_class_mixup`, `train`, `test`) now follow directly after the `SynerMixEfficientNet` class. Its content reappears inside `main()` in Step 6.

- [ ] **Step 4: Parameterize `train()` and `test()`**

Change the `train` signature from:
```python
def train(opt, train_loader, model, criterion, optimizer, epoch, use_cuda):
```
to:
```python
def train(opt, train_loader, train_dataset, model, criterion, optimizer, epoch, use_cuda, device):
```

Inside `train()`:
- `supplement_batch(inputs, targets, train_dataset, min_samples=opt.synermix_min_samples)` now refers to the parameter (no code change needed — the name matches).
- Replace the feature-extraction block (the one with two debug prints) with:

```python
            features_by_class = {}
            for cls in unique_classes:
                cls_idx = torch.where(targets == cls)[0]
                if len(cls_idx) >= 2:  # Only process classes with at least 2 samples
                    cls_inputs = inputs[cls_idx]
                    # Extract features for this class (gradients flow through)
                    features = model.extract_features(cls_inputs)
                    if len(features.shape) > 2:
                        features = features.view(features.size(0), -1)
                    features_by_class[cls.item()] = features
```

(This also removes the inner `batch_size = features.size(0)` line that shadowed the outer `batch_size`.)

Change the `test` signature from:
```python
def test(opt, val_loader, model, criterion, epoch, use_cuda):
    global best_acc
```
to:
```python
def test(opt, val_loader, model, criterion, epoch, use_cuda, device):
```
(The `global best_acc` declaration is removed — `test()` never touches `best_acc`.)

- [ ] **Step 5: Remove the debug print in `intra_class_mixup`**

Replace:
```python
            # Debug shapes
            print(f"Features shape: {features.shape}, Weights shape: {weights.shape}")
            
            # Perform weighted sum along sample dimension
            mixed_feature = (features * weights).sum(dim=0, keepdim=True)
```
with:
```python
            # Perform weighted sum along sample dimension
            mixed_feature = (features * weights).sum(dim=0, keepdim=True)
```

Also in the same function, replace the shape-consistency block:
```python
            shapes = [f.shape for f in mixed_features]
            if len(set(shapes)) > 1:
                print(f"Warning: Mixed features have inconsistent shapes: {shapes}")
                # Reshape all features to have the same shape as the first one
                target_shape = mixed_features[0].shape
                for i in range(1, len(mixed_features)):
                    if mixed_features[i].shape != target_shape:
                        print(f"Reshaping feature {i} from {mixed_features[i].shape} to {target_shape}")
                        mixed_features[i] = mixed_features[i].view(target_shape)
```
with:
```python
            shapes = [f.shape for f in mixed_features]
            if len(set(shapes)) > 1:
                target_shape = mixed_features[0].shape
                for i in range(1, len(mixed_features)):
                    if mixed_features[i].shape != target_shape:
                        mixed_features[i] = mixed_features[i].view(target_shape)
```

- [ ] **Step 6: Add `main()` at the end of the file**

Append after the last function definition (`test`), replacing the orphaned `# Training loop` block currently stranded inside `test()` (delete those stranded lines — everything from `    # Training loop` through the final `    print(f'Best accuracy: {best_acc:.2f}%')`):

```python
def main():
    # Create base model and wrap it for feature extraction
    base_model = EfficientNet.from_name(opt.arch, num_classes=opt.classes,
                                       override_params={'dropout_rate': opt.dropout, 'drop_connect_rate': opt.dropconnect})
    model = SynerMixEfficientNet(base_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if device.type == 'cuda':
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

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-4)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler_cosine)

    os.makedirs(opt.checkpoint, exist_ok=True)

    # Resume functionality
    if opt.resume:
        print('==> Resuming from checkpoint..')
        checkpoint_dir = os.path.dirname(opt.resume)
        resume = torch.load(opt.resume)
        best_acc = resume['best_acc']
        start_epoch = resume['epoch']
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        logger = Logger(os.path.join(checkpoint_dir, 'log.txt'), resume=True)
    else:
        start_epoch = opt.start_epoch
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'))
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Train AUROC.', 'Valid AUROC.'])

    # Training loop
    for epoch in range(start_epoch, opt.epochs):
        opt.lr = optimizer.state_dict()['param_groups'][0]['lr']

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.lr))
        if epoch >= opt.synermix_warmup_epochs:
            print('SynerMix Beta: %.2f' % opt.synermix_beta)

        train_loss, train_acc, train_auroc = train(opt, train_loader, train_dataset, model, criterion, optimizer, epoch, use_cuda, device)
        test_loss, test_acc, test_auroc = test(opt, val_loader, model, criterion, epoch, use_cuda, device)

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


if __name__ == '__main__':
    main()
```

Note two intentional behavior fixes vs. the old dead loop: the loop now starts from the resumed `start_epoch` (previously it always used `opt.start_epoch`, silently ignoring resume), and `opt.checkpoint` is created if missing.

- [ ] **Step 7: Verify the fix — rerun the smoke command**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe synermix_pretrain.py --source_dataset dataset/_smoke --epochs 2 --synermix_warmup_epochs 1 --train_batch 4 --test_batch 8 --num_workers 0 --size 64 --checkpoint ./log/smoke
```
Expected: `Epoch: [1 | 2]` (standard/CutMix warmup path), then `Epoch: [2 | 2]` with `Epoch 2: Using SynerMix with beta=0.80` (SynerMix path — exercises `supplement_batch`, `intra_class_mixup`, `inter_class_mixup`), `Validation: ...` lines, `Best accuracy: ...`, exit 0. Then:
```powershell
Test-Path log/smoke/checkpoint.pth.tar; Test-Path log/smoke/model_best.pth.tar
```
Expected: `True` / `True`

- [ ] **Step 8: Verify import-safety is preserved**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: `All tests passed successfully!` (importing `synermix_pretrain` must NOT start training).

- [ ] **Step 9: Commit**

```powershell
git add synermix_pretrain.py
git commit -m "fix: make SynerMix training loop reachable via main() and drop debug prints"
```

---

### Task 4: Fix quick_start.py's broken import

`quick_start.py:13` does `from efficientnet import EfficientNet` — no such module exists anywhere (verified: `ModuleNotFoundError`). The script cannot run at all.

**Files:**
- Modify: `quick_start.py:13`

- [ ] **Step 1: Demonstrate the failure**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe quick_start.py
```
Expected: `ModuleNotFoundError: No module named 'efficientnet'`

- [ ] **Step 2: Fix the import**

In `quick_start.py`, replace:
```python
from efficientnet import EfficientNet
```
with:
```python
from models import EfficientNet
```

- [ ] **Step 3: Verify with a real run (bundled weights + smoke test split)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe quick_start.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --test_batch 16 --num_workers 0 --size 64 --resume weights/pre-train/efficientnet/stargan.pth.tar
```
Expected: `=> using pre-trained model 'weights/pre-train/efficientnet/stargan.pth.tar'` (the GroupNorm state dict loads cleanly into `models/`'s architecture), `Performance of dataset/_smoke` and two `... | Loss:... | AUROC:...` result lines, exit 0. (`--test_batch 16` puts all 16 test images in one batch so per-batch AUROC always sees both classes.)

- [ ] **Step 4: Commit**

```powershell
git add quick_start.py
git commit -m "fix: import EfficientNet from models package in quick_start"
```

---### Task 5: Add argument types to options/transfer.py

`options/transfer.py` declares numeric arguments without `type=` — argparse then delivers CLI overrides as STRINGS (`--epochs 1` → `"1"`), crashing `range(start_epoch, opt.epochs)` and every arithmetic use. Defaults happen to be ints, which is why it only breaks when someone passes flags. Task 6's smoke run needs this fixed first.

**Files:**
- Modify: `options/transfer.py:18-44`

- [ ] **Step 1: Demonstrate the failure**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe -c "import sys; sys.argv=['t','--target','style2','--epochs','1']; from options.transfer import BaseOptions; o=BaseOptions().parse(print_options=False); print(repr(o.epochs))"
```
Expected: `'1'` (a string — the bug).

- [ ] **Step 2: Add types**

In `options/transfer.py`, replace the argument block from `parser.add_argument('--classes', default=2)` through `parser.add_argument('--gpu_id', default=0)` with:

```python
        parser.add_argument('--classes', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--iterations', type=int, default=500)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--train_batch', type=int, default=200)
        parser.add_argument('--test_batch', type=int, default=200)
        parser.add_argument('--lr', type=float, default=0.04)
        parser.add_argument('--schedule', nargs='+', type=int, default=[250, 250])
        parser.add_argument('--momentum', type=float, default=0.1)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--s', type=float, default=1.0)

        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--manual_seed', type=int, default=7)
        parser.add_argument('--size', type=int, default=128)

        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
        parser.add_argument('--dropconnect', type=float, default=0.2, help='Dropconnect probability')

        parser.add_argument('--cm_prob', type=float, default=0.5, help='Cutmix probability')
        parser.add_argument('--cm_beta', type=float, default=1.0)
        parser.add_argument('--blur_prob', type=float, default=0.5, help='Gaussian probability')
        parser.add_argument('--blog_sig', type=float, default=0.5, help='Gaussian sigma')
        parser.add_argument('--jpg_prob', type=float, default=0.5, help='JPEG compression')
        parser.add_argument('--fc_name', type=str, default='_fc.')

        parser.add_argument('--gpu_id', type=int, default=0)
```

(Keep the surrounding lines — `--target`, `--source_dataset`, `--target_dataset`, `--mode`, `--arch`, `--checkpoint` above, and `--pretrained_dir`, `--resume` below — unchanged.)

- [ ] **Step 3: Verify**

Rerun the Step 1 command. Expected: `1` (an int).

- [ ] **Step 4: Commit**

```powershell
git add options/transfer.py
git commit -m "fix: add argparse types to numeric transfer options"
```

---

### Task 6: Fix transfer.py — wrong package import plus four bugs

`transfer.py` imports the BatchNorm EfficientNet (`from EfficientNet.model_pytorch import EfficientNet`) whose `_bn*` state-dict keys cannot load the bundled `_gn*` weights — the script is unusable with this repo's weights. Plus: a `NameError` on fresh (non-resume) runs, an AUROC meter that is never updated, a wrong loader in a log line, and tensor-vs-float meter updates.

**Files:**
- Modify: `transfer.py`

- [ ] **Step 1: Demonstrate the state-dict incompatibility**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe transfer.py --target style2 --source_dataset dataset/_smoke --target_dataset dataset/_smoke --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar --epochs 1 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_transfer
```
Expected: `RuntimeError: Error(s) in loading state_dict for EfficientNet: Missing key(s) ... _bn0.weight ...` (BatchNorm keys expected, GroupNorm keys provided).

- [ ] **Step 2: Fix the import**

In `transfer.py`, replace:
```python
from EfficientNet.model_pytorch import EfficientNet
```
with:
```python
from models import EfficientNet
```

- [ ] **Step 3: Fix the fresh-run NameError in checkpoint saving**

In the epoch loop at the bottom, replace:
```python
    }, is_best, checkpoint=checkpoint)
```
with:
```python
    }, is_best, checkpoint=opt.checkpoint)
```
(`checkpoint` is only defined inside the `if opt.resume:` branch; every fresh run crashed with `NameError` at the end of epoch 1.)

- [ ] **Step 4: Update the train AUROC meter and use float meter updates**

In `train()`, replace:
```python
        # measure accuracy and record loss
        losses.update(loss.data.tolist(), inputs.size(0))
        cls_losses.update(loss_cls, inputs.size(0))
        sp_losses.update(loss_sp, inputs.size(0))
        main_losses.update(loss_main.tolist(), inputs.size(0))
        alpha.update(sp_gamma, inputs.size(0))
```
with:
```python
        # measure accuracy and record loss
        try:
            auroc = roc_auc_score(targets.cpu().detach().numpy(),
                                  outputs.cpu().detach().numpy()[:, 1])
            arc.update(auroc, inputs.size(0))
        except ValueError:
            pass  # single-class batch; AUROC undefined
        losses.update(loss.data.tolist(), inputs.size(0))
        cls_losses.update(loss_cls.item(), inputs.size(0))
        sp_losses.update(loss_sp.item(), inputs.size(0))
        main_losses.update(loss_main.tolist(), inputs.size(0))
        alpha.update(sp_gamma.item(), inputs.size(0))
```
(The `arc` meter was logged every epoch — including into `log.txt` as Train AUROC — but never updated, so it always read 0.)

- [ ] **Step 5: Reconcile the validation loss and fix the wrong loader in test()'s log line**

In `test()`, replace:
```python
            loss = loss_main + 0*loss_sp + 0*loss_cls
```
with:
```python
            # Regularizers are reported via their meters but excluded from the
            # validation loss by design (they don't measure generalization).
            loss = loss_main
```

In the same function, replace:
```python
            cls_losses.update(loss_cls, inputs.size(0))
            sp_losses.update(loss_sp, inputs.size(0))
```
with:
```python
            cls_losses.update(loss_cls.item(), inputs.size(0))
            sp_losses.update(loss_sp.item(), inputs.size(0))
```

And in its final print statement, replace:
```python
                     batch=batch_idx+1, size=len(train_loader), loss=losses.avg, main=main_losses.avg, sp=sp_losses.avg, cls=cls_losses.avg, ac=arc.avg))
```
with:
```python
                     batch=batch_idx+1, size=len(val_loader), loss=losses.avg, main=main_losses.avg, sp=sp_losses.avg, cls=cls_losses.avg, ac=arc.avg))
```

- [ ] **Step 6: Smoke-run transfer end-to-end**

Rerun the Step 1 command. Expected: weights load silently, one epoch runs, `Train | 4/4 | Loss:... | ... | AUROC:...` with a nonzero AUROC, two `Test | 1/1 | ...` lines (target + source val), checkpoint saved WITHOUT `NameError`, exit 0. Verify:
```powershell
Test-Path log/smoke_transfer/checkpoint.pth.tar
```
Expected: `True`

- [ ] **Step 7: Commit**

```powershell
git add transfer.py
git commit -m "fix: point transfer at GroupNorm models package and repair meters, val loss, checkpoint save"
```

---

### Task 7: Delete the duplicate EfficientNet package

After Task 6, nothing imports `EfficientNet.model_pytorch`. The directory also contains a training notebook (worth keeping — move to `notebooks/`) and a stale `EfficientNet/utils/` copy (delete with the rest).

**Files:**
- Move: `EfficientNet/train.ipynb` → `notebooks/efficientnet_train.ipynb`
- Delete: `EfficientNet/` (entire directory)
- Modify: `docs/superpowers/specs/2026-06-10-tgd-portfolio-sprint-design.md` (resnext correction)

- [ ] **Step 1: Confirm no production imports remain**

```powershell
git grep -n "EfficientNet.model_pytorch" -- "*.py"
```
Expected: no output (exit code 1 from git grep means no matches — that is the pass condition).

- [ ] **Step 2: Move the notebook, delete the package**

```powershell
git mv EfficientNet/train.ipynb notebooks/efficientnet_train.ipynb
git rm -r EfficientNet/
```
(The notebook may reference the deleted package internally; it is an archived artifact, acceptable.)

- [ ] **Step 3: Verify nothing broke**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe test_synermix.py
```
Expected: `All tests passed successfully!`

- [ ] **Step 4: Amend the spec — resnext stays**

The spec says to delete `resnext/` as unused; the audit was wrong — `gradio_ui.py` instantiates `resnext50_32x4d` at lines 108, 264, and 606. In `docs/superpowers/specs/2026-06-10-tgd-portfolio-sprint-design.md`, replace:

```
3. **Consolidate model packages.** All imports point at `models/`; delete `EfficientNet/` (duplicate package) and `resnext/` (unused). Check `gradio_ui.py`, `quick_start.py`, notebooks for stragglers before deleting.
```
with:
```
3. **Consolidate model packages.** All imports point at `models/`; delete `EfficientNet/` (BatchNorm variant — its state-dict keys are incompatible with the bundled GroupNorm weights). `resnext/` stays: `gradio_ui.py` instantiates `resnext50_32x4d` in ensemble mode. Check `gradio_ui.py`, `quick_start.py`, notebooks for stragglers before deleting.
```

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "refactor: remove duplicate BatchNorm EfficientNet package, keep models/ as canonical"
```

---

### Task 8: Full entry-point smoke matrix

Final verification that every entry point runs on CPU. Three scripts not yet smoke-tested: `eval.py`, `pretrain.py`, `early_stop_pretrain.py`. No code changes are expected; if any command fails, that is a finding to fix before closing Phase 1.

**Files:** none (verification only)

- [ ] **Step 1: eval.py (bundled pretrain + T-GD weights)**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe eval.py --source_dataset dataset/_smoke --target_dataset dataset/_smoke --pretrained_dir weights/pre-train/efficientnet/stargan.pth.tar --resume weights/t-gd/efficientnet/star_to_style2.pth.tar --test_batch 16 --num_workers 0 --size 64
```
Expected: weights load, two `Performance of dataset/_smoke` blocks with `Loss:`/`AUROC:` values, exit 0.

- [ ] **Step 2: pretrain.py**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe pretrain.py --source_dataset dataset/_smoke --epochs 2 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_pretrain
```
Expected: 2 epochs of train/validation output, `log/smoke_pretrain/checkpoint.pth.tar` exists, exit 0.

- [ ] **Step 3: early_stop_pretrain.py**

```powershell
$env:PYTHONUTF8='1'; .\venv\Scripts\python.exe early_stop_pretrain.py --source_dataset dataset/_smoke --epochs 2 --train_batch 4 --test_batch 16 --num_workers 0 --size 64 --checkpoint ./log/smoke_es
```
Expected: 2 epochs (or early stop) of output, checkpoint written to `log/smoke_es/`, exit 0.

- [ ] **Step 4: Confirm clean tree and report**

```powershell
git status --short
```
Expected: empty output (everything committed; `dataset/` and `log/` are gitignored). Report any deviations found in Steps 1–3 rather than papering over them.
