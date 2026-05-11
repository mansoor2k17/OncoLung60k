"""
Verify that the split CSV has no patient leakage between folds.

Usage:
    python scripts/verify_no_leakage.py --csv splits/oncolung60k_5fold.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.splits import verify_no_patient_leakage, summarise_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Folds:    {sorted(df['fold'].unique())}")

    print("\n--- Per-fold summary ---")
    summarise_splits(df)

    print("\n--- Patient leakage check ---")
    verify_no_patient_leakage(df)


if __name__ == "__main__":
    main()
