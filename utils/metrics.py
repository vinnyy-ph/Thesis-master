"""Binary-classification metrics, computed once per evaluation pass.

Replaces ad-hoc per-batch AUROC. Accumulate y_true / y_score across all
batches, then call compute_metrics ONCE — a single-class *batch* can no
longer corrupt the score. Unlike sklearn's per-call behaviour on
single-class input (which returns NaN), ranking metrics degrade to None.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_score):
    """Binary metrics from true labels and positive-class probabilities.

    Args:
        y_true: 1-D array-like of 0/1 integer labels.
        y_score: 1-D array-like of positive-class (class 1) probabilities in
            [0, 1] — e.g. F.softmax(outputs, dim=1)[:, 1].

    Returns:
        dict: accuracy, precision, recall, f1, average_precision, auroc,
        confusion_matrix ({tn, fp, fn, tp}). auroc and average_precision are
        None when y_true has a single class. Never returns NaN.

    Raises:
        ValueError: if y_true is empty or contains labels outside {0, 1}.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0:
        raise ValueError("compute_metrics: y_true is empty")
    labels_present = set(np.unique(y_true).tolist())
    if not labels_present.issubset({0, 1}):
        raise ValueError(
            f"compute_metrics expects binary labels in {{0,1}}, got {sorted(labels_present)}"
        )

    y_pred = (y_score >= 0.5).astype(int)
    both = labels_present == {0, 1}
    auroc = float(roc_auc_score(y_true, y_score)) if both else None
    ap = float(average_precision_score(y_true, y_score)) if both else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'average_precision': ap,
        'auroc': auroc,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }
