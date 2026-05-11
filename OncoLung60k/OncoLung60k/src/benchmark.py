"""
Multi-model SOTA benchmark.

Trains all baselines under the identical patient-wise k-fold CV protocol
and produces a final results table.

Usage:
    python -m src.benchmark \
        --configs_dir configs/ \
        --data_dir data/oncolung60k \
        --output_dir runs/benchmark
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# All baselines benchmarked in the paper
BENCHMARK_CONFIGS = [
    "oncolung_resnet50.yaml",
    "oncolung_densenet121.yaml",
    "oncolung_efficientnet_b0.yaml",
    "oncolung_convnext_tiny.yaml",
    "oncolung_convnext_small.yaml",
    "oncolung_convnext_base.yaml",
    "oncolung_vit_base.yaml",
    "oncolung_swin_base.yaml",
    "oncolung_modified_convnext.yaml",
]


def main(args):
    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_name in BENCHMARK_CONFIGS:
        config_path = configs_dir / config_name
        if not config_path.exists():
            print(f"[SKIP] Config not found: {config_path}")
            continue

        run_name = config_path.stem
        run_dir = output_dir / run_name
        if (run_dir / "cv_summary.csv").exists() and not args.force:
            print(f"[CACHED] {run_name} already complete, skipping")
        else:
            print(f"\n[RUNNING] {run_name}")
            cmd = [
                sys.executable, "-m", "src.kfold_cv",
                "--config", str(config_path),
                "--data_dir", args.data_dir,
                "--output_dir", str(run_dir),
            ]
            subprocess.run(cmd, check=True)

        # Read summary
        summary = pd.read_csv(run_dir / "cv_summary.csv", index_col=0)
        per_fold = pd.read_csv(run_dir / "per_fold_metrics.csv")
        row = {
            "model": run_name,
            "accuracy_mean": per_fold["accuracy"].mean(),
            "accuracy_std": per_fold["accuracy"].std(),
            "f1_mean": per_fold["f1_macro"].mean(),
            "f1_std": per_fold["f1_macro"].std(),
            "auc_mean": per_fold["auc_macro_ovr"].mean(),
            "auc_std": per_fold["auc_macro_ovr"].std(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "benchmark_summary.csv", index=False)
    print("\n========== BENCHMARK SUMMARY ==========")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--force", action="store_true",
                    help="Re-run even if cached results exist")
    args = p.parse_args()
    main(args)
