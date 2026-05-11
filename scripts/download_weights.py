"""
Helper: Download OncoLung60K dataset or pretrained weights from Zenodo.

Usage:
    python scripts/download_weights.py --output checkpoints/
    python scripts/download_dataset.py --output data/
"""
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_URL = "https://zenodo.org/records/14995223/files/"

DATASET_FILES = {
    "OncoLung60K_patches.tar.gz": "OncoLung60K_patches.tar.gz",
}

WEIGHT_FILES = {
    "fold0_best.pt": "modified_convnext_fold0_best.pt",
    "fold1_best.pt": "modified_convnext_fold1_best.pt",
    "fold2_best.pt": "modified_convnext_fold2_best.pt",
    "fold3_best.pt": "modified_convnext_fold3_best.pt",
    "fold4_best.pt": "modified_convnext_fold4_best.pt",
}


def download(filename: str, output_dir: Path) -> None:
    """Download one file from Zenodo with progress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    url = ZENODO_URL + filename
    out_path = output_dir / filename

    if out_path.exists():
        print(f"  [SKIP] {out_path} already exists")
        return

    def hook(blocknum, blocksize, totalsize):
        downloaded = blocknum * blocksize
        if totalsize > 0:
            pct = min(100, downloaded * 100 / totalsize)
            sys.stdout.write(f"\r  {filename}: {pct:5.1f}% "
                             f"({downloaded/1e6:.1f}/{totalsize/1e6:.1f} MB)")
            sys.stdout.flush()

    print(f"  Downloading {filename}...")
    urlretrieve(url, out_path, hook)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path,
                    help="Output directory")
    p.add_argument("--type", choices=["weights", "dataset"], default="weights")
    args = p.parse_args()

    print(f"Downloading {args.type} from Zenodo to {args.output}/")
    files = WEIGHT_FILES if args.type == "weights" else DATASET_FILES
    for local, remote in files.items():
        download(remote, args.output)
    print(f"\nAll {args.type} files downloaded to {args.output}/")


if __name__ == "__main__":
    main()
