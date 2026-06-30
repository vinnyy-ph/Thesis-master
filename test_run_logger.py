"""Standalone test for utils.run_logger (no pytest; run with python)."""
import json
import os
import shutil
import tempfile
from argparse import Namespace

from utils import run_logger


def test_start_run_writes_config():
    root = tempfile.mkdtemp()
    try:
        opt = Namespace(seed=7, arch='efficientnet-b0', resume='')
        run_dir = run_logger.start_run(opt, root=root)
        assert os.path.isdir(run_dir)
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        assert cfg['seed'] == 7 and cfg['arch'] == 'efficientnet-b0'
    finally:
        shutil.rmtree(root)


def test_epoch_logging_and_finalize():
    root = tempfile.mkdtemp()
    try:
        run_dir = run_logger.start_run(Namespace(seed=1), root=root)
        m1 = {'auroc': 0.8, 'confusion_matrix': {'tn': 1, 'fp': 0, 'fn': 0, 'tp': 1}}
        m2 = {'auroc': 0.9, 'confusion_matrix': {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}}
        run_logger.log_epoch(run_dir, 1, m1)
        run_logger.log_epoch(run_dir, 2, m2)
        run_logger.finalize(run_dir, m2)
        with open(os.path.join(run_dir, 'metrics.json')) as f:
            data = json.load(f)
        assert len(data['per_epoch']) == 2
        assert data['final']['auroc'] == 0.9
        assert data['final']['confusion_matrix_tp'] == 2  # flattened
        assert os.path.exists(os.path.join(run_dir, 'metrics.csv'))
    finally:
        shutil.rmtree(root)


def test_finalize_without_epochs():
    root = tempfile.mkdtemp()
    try:
        run_dir = run_logger.start_run(Namespace(seed=1), root=root)
        run_logger.finalize(run_dir, {'auroc': 0.7, 'accuracy': 0.6})
        with open(os.path.join(run_dir, 'metrics.json')) as f:
            data = json.load(f)
        assert data['per_epoch'] == []
        assert data['final']['auroc'] == 0.7
        assert os.path.exists(os.path.join(run_dir, 'metrics.csv'))
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_start_run_writes_config()
    test_epoch_logging_and_finalize()
    test_finalize_without_epochs()
    print("All run_logger tests passed.")
