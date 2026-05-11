"""
Tissue masking and patch extraction.

For raw microscope-captured histopathology images:
  1. Convert RGB -> HSV.
  2. Otsu threshold on the saturation channel -> tissue mask.
  3. Morphological closing to fill internal holes.
  4. Non-overlapping 512x512 tile extraction within the tissue mask.

Usage:
    python -m src.data.preprocessing \
        --input raw_images/ --output patches/ --tile_size 512
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def get_tissue_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a binary tissue mask from an H&E image (BGR uint8).

    Args:
        image_bgr: H x W x 3 uint8 array (OpenCV BGR).

    Returns:
        H x W uint8 binary mask (255 = tissue, 0 = background).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    _, mask = cv2.threshold(saturation, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def extract_tiles(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 512,
    min_tissue_fraction: float = 0.5,
):
    """Extract non-overlapping tiles where mask coverage exceeds threshold.

    Yields:
        (y, x, tile_bgr) tuples.
    """
    H, W = mask.shape
    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            tile_mask = mask[y:y + tile_size, x:x + tile_size]
            if tile_mask.mean() / 255.0 < min_tissue_fraction:
                continue
            tile = image_bgr[y:y + tile_size, x:x + tile_size]
            yield y, x, tile


def process_directory(input_dir: Path, output_dir: Path, tile_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total, n_kept = 0, 0
    for img_path in sorted(input_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            continue
        mask = get_tissue_mask(img)
        for y, x, tile in extract_tiles(img, mask, tile_size):
            n_total += 1
            out_name = f"{img_path.stem}_y{y:05d}_x{x:05d}.jpg"
            cv2.imwrite(str(output_dir / out_name), tile,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_kept += 1
    print(f"Saved {n_kept} tiles from {input_dir} to {output_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tile_size", type=int, default=512)
    args = p.parse_args()
    process_directory(args.input, args.output, args.tile_size)


if __name__ == "__main__":
    main()
