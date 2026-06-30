"""Per-run artifacts: config + metrics to runs/<timestamp>/."""
import csv
import json
import os
from datetime import datetime


def start_run(opt, root='runs'):
    """Create runs/<timestamp>/ and dump the full options namespace.

    Returns the run directory path. Call once at the start of a run.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({k: _jsonable(v) for k, v in vars(opt).items()}, f, indent=2)
    return run_dir


def log_epoch(run_dir, epoch, metrics):
    """Append one epoch's (flattened) metrics to the per-epoch record."""
    path = os.path.join(run_dir, '_epochs.json')
    epochs = []
    if os.path.exists(path):
        with open(path) as f:
            epochs = json.load(f)
    epochs.append({'epoch': epoch, **_flatten(metrics)})
    with open(path, 'w') as f:
        json.dump(epochs, f, indent=2)


def finalize(run_dir, final_metrics):
    """Write metrics.json ({final, per_epoch}) and metrics.csv."""
    epochs_path = os.path.join(run_dir, '_epochs.json')
    per_epoch = []
    if os.path.exists(epochs_path):
        with open(epochs_path) as f:
            per_epoch = json.load(f)
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump({'final': _flatten(final_metrics), 'per_epoch': per_epoch}, f, indent=2)
    rows = per_epoch if per_epoch else [{'epoch': 0, **_flatten(final_metrics)}]
    fieldnames = sorted({k for r in rows for k in r})
    with open(os.path.join(run_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _flatten(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub, subv in v.items():
                out[f'{k}_{sub}'] = subv
        else:
            out[k] = v
    return out


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
