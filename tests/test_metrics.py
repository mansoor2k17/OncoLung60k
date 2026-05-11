"""Unit tests for metric computation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils.metrics import (
    compute_all_metrics,
    per_class_clinical_metrics,
    paired_significance_tests,
    bonferroni_alpha,
)


def test_perfect_classifier():
    n = 200
    y = np.random.randint(0, 4, size=n)
    probs = np.eye(4)[y]  # one-hot, perfect predictions
    m = compute_all_metrics(y, probs, n_classes=4)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["auc_macro_ovr"] == 1.0


def test_random_classifier_around_chance():
    np.random.seed(0)
    n = 4000
    y = np.random.randint(0, 4, size=n)
    probs = np.random.dirichlet([1, 1, 1, 1], size=n)
    m = compute_all_metrics(y, probs, n_classes=4)
    assert 0.20 <= m["accuracy"] <= 0.30


def test_clinical_metrics_shape():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 2, 3, 0])
    df = per_class_clinical_metrics(y, pred, n_classes=4,
                                     class_names=["A", "B", "C", "D"])
    assert "macro_avg" in df.index
    assert (df.loc["A", "sensitivity"] == 0.5)


def test_paired_significance():
    a = np.array([0.91, 0.92, 0.90, 0.91, 0.93])
    b = np.array([0.85, 0.86, 0.84, 0.85, 0.87])
    res = paired_significance_tests(a, b)
    assert res["mean_diff"] > 0
    assert res["t_pvalue"] < 0.05  # strong difference


def test_bonferroni():
    assert bonferroni_alpha(36, 0.05) == pytest.approx(0.05 / 36)


import pytest
