"""
Single training run.

Usage:
    python -m src.train --config configs/oncolung_modified_convnext.yaml \
                         --data_dir data/oncolung60k \
                         --output_dir runs/exp1
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
from src.utils.metrics import compute_all_metrics


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        all_logits.append(model(x).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    return compute_all_metrics(y_true, probs, n_classes)


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    yaml.safe_dump(cfg, (output_dir / "config.yaml").open("w"))

    # Load split CSV
    df = pd.read_csv(cfg["splits_csv"])
    test_fold = cfg.get("test_fold", 0)
    train_df = df[df["fold"] != test_fold].reset_index(drop=True)
    test_df = df[df["fold"] == test_fold].reset_index(drop=True)
    print(f"Train: {len(train_df)} patches | Test: {len(test_df)} patches")
    print(f"Train patients: {train_df['patient_id'].nunique()} | "
          f"Test patients: {test_df['patient_id'].nunique()}")

    # Datasets and loaders
    image_size = cfg.get("image_size", 256)
    train_tf = build_transforms(image_size, is_training=True)
    eval_tf = build_transforms(image_size, is_training=False)

    train_set = HistoPathologyDataset(train_df, args.data_dir, transform=train_tf)
    test_set = HistoPathologyDataset(test_df, args.data_dir, transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg.get("num_workers", 4),
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg.get("num_workers", 4),
                             pin_memory=True)

    # Model, optimiser, scheduler
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

    # Training loop
    best_acc = 0.0
    history = []
    patience_counter = 0
    patience = cfg.get("early_stop_patience", 10)

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        scheduler.step()
        metrics = evaluate(model, test_loader, device, cfg["num_classes"])

        log = {"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
               **{k: v for k, v in metrics.items()
                  if isinstance(v, (int, float))}}
        history.append(log)
        print(f"Ep {epoch+1:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | "
              f"val_acc {metrics['accuracy']:.4f} | "
              f"val_auc {metrics['auc_macro_ovr']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save({"model": model.state_dict(), "metrics": metrics, "epoch": epoch},
                       output_dir / "best.pt")
            with (output_dir / "best_metrics.json").open("w") as f:
                json.dump({k: (v if isinstance(v, (int, float, list))
                               else str(v)) for k, v in metrics.items()}, f, indent=2)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    main(args)
