"""Standalone test for utils.metrics (no pytest; run with python)."""
import math

from utils.metrics import compute_metrics


def _assert_no_nan(m):
    for k, v in m.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"{k} is NaN"


def test_perfect_separation():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m['auroc'] == 1.0
    assert m['accuracy'] == 1.0
    assert m['confusion_matrix'] == {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}
    _assert_no_nan(m)


def test_confusion_matrix_mixed():
    # preds: 0.1->0, 0.9->1, 0.2->0, 0.8->1  vs true 0,0,1,1
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert m['confusion_matrix'] == {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1}
    assert abs(m['accuracy'] - 0.5) < 1e-9


def test_single_class_returns_none_not_nan():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert m['auroc'] is None
    assert m['average_precision'] is None
    assert m['accuracy'] == 1.0  # all predicted positive, all truly positive
    _assert_no_nan(m)


def test_non_binary_raises():
    try:
        compute_metrics([0, 1, 2], [0.1, 0.5, 0.9])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-binary labels")


def test_empty_raises():
    try:
        compute_metrics([], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty input")


if __name__ == '__main__':
    test_perfect_separation()
    test_confusion_matrix_mixed()
    test_single_class_returns_none_not_nan()
    test_non_binary_raises()
    test_empty_raises()
    print("All metrics tests passed.")
