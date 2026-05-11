"""
Patient-wise k-fold cross-validation splits.

CRITICAL: Histopathology datasets contain many patches per patient.
Image-level random splits leak patient information between train and test
and inflate performance estimates dramatically. We use
StratifiedGroupKFold with patient_id as the grouping variable to
guarantee zero patient overlap between folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_patient_wise_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a 'fold' column with patient-wise stratified k-fold assignment.

    Args:
        df: DataFrame with at least [label, patient_id] columns.
        n_splits: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with an additional 'fold' column (0..n_splits-1).
    """
    if "fold" in df.columns:
        df = df.drop(columns=["fold"])

    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1

    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(df, df["label"], groups=df["patient_id"])
    ):
        df.loc[test_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Some rows were not assigned to a fold"
    return df


def verify_no_patient_leakage(df: pd.DataFrame) -> None:
    """Assert that no patient_id appears in more than one fold."""
    if "fold" not in df.columns:
        raise ValueError("DataFrame has no 'fold' column")

    overlaps = []
    folds = sorted(df["fold"].unique())
    for i, fold_a in enumerate(folds):
        patients_a = set(df.loc[df["fold"] == fold_a, "patient_id"])
        for fold_b in folds[i + 1:]:
            patients_b = set(df.loc[df["fold"] == fold_b, "patient_id"])
            shared = patients_a & patients_b
            if shared:
                overlaps.append((fold_a, fold_b, sorted(shared)))

    if overlaps:
        msg = "\n".join(
            f"Folds {a} and {b} share patients: {p}" for a, b, p in overlaps
        )
        raise AssertionError(f"Patient leakage detected:\n{msg}")
    print(f"OK: No patient leakage across {len(folds)} folds")


def get_train_test_dfs(df: pd.DataFrame, test_fold: int):
    """Split the DataFrame into train and test using the 'fold' column."""
    test_df = df[df["fold"] == test_fold].reset_index(drop=True)
    train_df = df[df["fold"] != test_fold].reset_index(drop=True)
    return train_df, test_df


def summarise_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Print a summary table of patches/patients/classes per fold."""
    rows = []
    for fold in sorted(df["fold"].unique()):
        f = df[df["fold"] == fold]
        rows.append({
            "fold": fold,
            "n_patches": len(f),
            "n_patients": f["patient_id"].nunique(),
            **{f"class_{c}": (f["label"] == c).sum()
               for c in sorted(df["label"].unique())},
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    df = make_patient_wise_kfold(df, n_splits=args.n_splits, seed=args.seed)
    verify_no_patient_leakage(df)
    summarise_splits(df)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved splits to {args.output_csv}")
