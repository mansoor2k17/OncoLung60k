"""
Evaluate a single trained model on a specified test split.

Usage:
    python -m src.evaluate \
        --weights runs/cv_modified_convnext/fold0/best.pt \
        --config configs/oncolung_modified_convnext.yaml \
        --data_dir data/oncolung60k \
        --test_fold 0 \
        --output evaluation.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.augmentation import build_transforms
from src.data.dataset import HistoPathologyDataset
from src.models.builder import build_model
from src.train import evaluate


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(cfg["splits_csv"])
    test_df = df[df["fold"] == args.test_fold].reset_index(drop=True)

    eval_tf = build_transforms(cfg.get("image_size", 256), is_training=False)
    test_set = HistoPathologyDataset(test_df, args.data_dir, transform=eval_tf)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg.get("num_workers", 4), pin_memory=True)

    model = build_model(cfg["model"]["name"],
                        num_classes=cfg["num_classes"],
                        pretrained=False).to(device)
    state = torch.load(args.weights, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    metrics = evaluate(model, test_loader, device, cfg["num_classes"])
    print(json.dumps({k: (v if isinstance(v, (int, float, list)) else str(v))
                       for k, v in metrics.items()}, indent=2))

    with open(args.output, "w") as f:
        json.dump({k: (v if isinstance(v, (int, float, list)) else str(v))
                    for k, v in metrics.items()}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--test_fold", type=int, required=True)
    p.add_argument("--output", default="evaluation.json")
    args = p.parse_args()
    main(args)
