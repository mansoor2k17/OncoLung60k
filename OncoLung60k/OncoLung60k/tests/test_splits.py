"""Unit tests for patient-wise splitting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.utils.splits import make_patient_wise_kfold, verify_no_patient_leakage


def make_synthetic(n_patients=20, patches_per=15, n_classes=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for pid in range(n_patients):
        cls = pid % n_classes
        for _ in range(patches_per):
            rows.append({
                "filepath": f"img_{pid}_{rng.randint(0, 1000)}.png",
                "label": cls,
                "patient_id": f"P{pid:03d}",
            })
    return pd.DataFrame(rows)


def test_no_leakage_after_split():
    df = make_synthetic()
    df_split = make_patient_wise_kfold(df, n_splits=5, seed=42)
    verify_no_patient_leakage(df_split)


def test_all_patients_assigned():
    df = make_synthetic(n_patients=20)
    df_split = make_patient_wise_kfold(df, n_splits=5, seed=42)
    assigned = df_split.groupby("patient_id")["fold"].nunique()
    assert (assigned == 1).all(), "Each patient should appear in exactly one fold"


def test_balanced_folds():
    df = make_synthetic(n_patients=40, patches_per=10)
    df_split = make_patient_wise_kfold(df, n_splits=5, seed=42)
    fold_sizes = df_split.groupby("fold").size()
    avg = fold_sizes.mean()
    # Allow ±20% deviation
    assert (fold_sizes >= 0.8 * avg).all() and (fold_sizes <= 1.2 * avg).all()


def test_leakage_detection_raises():
    df = make_synthetic()
    df["fold"] = 0  # all in same fold = no leakage
    verify_no_patient_leakage(df)

    df["fold"] = [0] * (len(df) // 2) + [1] * (len(df) - len(df) // 2)
    # Some patient is now in both folds
    with pytest.raises(AssertionError):
        verify_no_patient_leakage(df)
