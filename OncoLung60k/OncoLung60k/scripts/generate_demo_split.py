"""
Generate a small SAMPLE split CSV showing the schema.

This is NOT the full dataset split. Replace splits/oncolung60k_5fold.csv
with the real CSV generated from your full dataset using:

    python -m src.utils.splits --input_csv your_master.csv \\
        --output_csv splits/oncolung60k_5fold.csv

This script only generates a 100-row demo file for quick smoke tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.splits import make_patient_wise_kfold

OUT_DIR = Path(__file__).resolve().parent.parent / "splits"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.RandomState(42)
CLASS_NAMES = ["adeno", "scc", "sclc", "normal"]

# Demo: 65 patients, ~15 patches per patient = ~975 rows
records = []
patient_class_map = {}
for pid in range(65):
    cls = pid % 4
    patient_class_map[pid] = cls
    n_patches = rng.randint(12, 18)
    for i in range(n_patches):
        records.append({
            "filepath": f"{CLASS_NAMES[cls]}/patient_{pid:02d}_y{rng.randint(100, 5000):05d}_x{rng.randint(100, 5000):05d}.jpg",
            "label": cls,
            "patient_id": f"P{pid:03d}",
        })

df = pd.DataFrame(records)
print(f"Generated demo dataset: {len(df)} patches, "
      f"{df['patient_id'].nunique()} patients")

df_split = make_patient_wise_kfold(df, n_splits=5, seed=42)
out_path = OUT_DIR / "oncolung60k_5fold_DEMO.csv"
df_split.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
print(f"Columns: {list(df_split.columns)}")
print()
print("Per-fold summary:")
for f in sorted(df_split["fold"].unique()):
    sub = df_split[df_split["fold"] == f]
    print(f"  Fold {f}: {len(sub)} patches, "
          f"{sub['patient_id'].nunique()} patients")
