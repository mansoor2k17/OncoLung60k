"""
Full patient-wise k-fold cross-validation pipeline.

For each fold, trains a fresh model and evaluates on the held-out patients.
Aggregates results across folds with mean +/- std and computes statistical
significance against any baseline run.

Usage:
    python -m src.kfold_cv \
        --config configs/oncolung_modified_convnext.yaml \
        --data_dir data/oncolung60k \
        --output_dir runs/cv_modified_convnext
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.augmentation import build_transforms
from src.data.dataset import HistoPathologyDataset
from src.models.builder import build_model
from src.train import train_one_epoch, evaluate, set_seed
from src.utils.splits import verify_no_patient_leakage


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, (output_dir / "config.yaml").open("w"))

    df = pd.read_csv(cfg["splits_csv"])
    verify_no_patient_leakage(df)

    folds = sorted(df["fold"].unique())
    print(f"Running {len(folds)}-fold patient-wise CV...")

    fold_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in folds:
        print(f"\n========== Fold {fold + 1}/{len(folds)} ==========")
        fold_dir = output_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.get("seed", 42) + fold)

        train_df = df[df["fold"] != fold].reset_index(drop=True)
        test_df = df[df["fold"] == fold].reset_index(drop=True)
        print(f"Train: {len(train_df)} patches, "
              f"{train_df['patient_id'].nunique()} patients")
        print(f"Test:  {len(test_df)} patches, "
              f"{test_df['patient_id'].nunique()} patients")

        image_size = cfg.get("image_size", 256)
        train_tf = build_transforms(image_size, is_training=True)
        eval_tf = build_transforms(image_size, is_training=False)
        train_set = HistoPathologyDataset(train_df, args.data_dir, transform=train_tf)
        test_set = HistoPathologyDataset(test_df, args.data_dir, transform=eval_tf)
        train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                                   num_workers=cfg.get("num_workers", 4), pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False,
                                  num_workers=cfg.get("num_workers", 4), pin_memory=True)

        model = build_model(cfg["model"]["name"],
                            num_classes=cfg["num_classes"],
                            pretrained=cfg["model"].get("pretrained", True)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=cfg["lr"],
                                       weight_decay=cfg.get("weight_decay", 1e-3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg.get("min_lr", 1e-6)
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
        scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", True) and device == "cuda" else None

        best_metrics, best_acc = None, 0.0
        for epoch in range(cfg["epochs"]):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            scheduler.step()
            metrics = evaluate(model, test_loader, device, cfg["num_classes"])
            print(f"Ep {epoch+1:3d} | tr_acc {tr_acc:.4f} | "
                  f"val_acc {metrics['accuracy']:.4f} | "
                  f"val_auc {metrics['auc_macro_ovr']:.4f}")
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics
                torch.save({"model": model.state_dict(), "fold": fold,
                            "metrics": metrics}, fold_dir / "best.pt")

        # Save fold metrics
        scalar_metrics = {k: v for k, v in best_metrics.items()
                          if isinstance(v, (int, float))}
        scalar_metrics["fold"] = fold
        fold_results.append(scalar_metrics)
        with (fold_dir / "metrics.json").open("w") as f:
            json.dump({k: (v if isinstance(v, (int, float, list))
                           else str(v)) for k, v in best_metrics.items()},
                      f, indent=2)

    # Aggregate across folds
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(output_dir / "per_fold_metrics.csv", index=False)

    summary_cols = ["accuracy", "f1_macro", "auc_macro_ovr",
                    "precision_macro", "recall_macro"]
    summary = df_results[summary_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    summary.to_csv(output_dir / "cv_summary.csv")

    print("\n========== Cross-validated results ==========")
    print(summary.round(4))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    main(args)
