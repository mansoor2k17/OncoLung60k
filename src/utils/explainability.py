"""
Grad-CAM, Grad-CAM++, and Score-CAM visualizations for the Modified ConvNeXt.

Includes:
  - Heatmap generation for arbitrary input images.
  - Quantitative IoU evaluation against pathologist-annotated ROIs.

Usage:
    python -m src.utils.explainability \
        --weights checkpoints/fold0_best.pt \
        --images sample_images/ \
        --output_dir gradcam_outputs/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["Adeno", "SCC", "SCLC", "Normal"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_target_layer(model):
    """Return the target layer for Grad-CAM on Modified ConvNeXt.

    For our architecture, we use the last ECB block in stage 4.
    """
    if hasattr(model, "stages"):
        return [model.stages[-1][-1]]
    if hasattr(model, "blocks"):
        return [model.blocks[-1]]
    raise ValueError("Could not infer target layer; please specify manually.")


def make_eval_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def generate_heatmaps(
    model,
    image_paths,
    output_dir: Path,
    device: str = "cuda",
    methods=("gradcam", "gradcam_pp", "scorecam"),
    image_size: int = 256,
):
    """Generate Grad-CAM-family heatmaps for a list of images.

    Saves: {output_dir}/{stem}_{method}_{class}.png
    """
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise ImportError(
            "Install grad-cam: `pip install grad-cam`"
        ) from e

    model.eval().to(device)
    target_layers = get_target_layer(model)
    tf = make_eval_transform(image_size)

    cam_classes = {
        "gradcam": GradCAM,
        "gradcam_pp": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
    }
    cams = {name: cam_classes[name](model=model, target_layers=target_layers)
            for name in methods if name in cam_classes}

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for img_path in image_paths:
        img_path = Path(img_path)
        pil = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        rgb = np.array(pil) / 255.0
        x = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else f"c{pred}"

        for name, cam in cams.items():
            grayscale = cam(input_tensor=x, targets=None)[0]
            viz = show_cam_on_image(rgb, grayscale, use_rgb=True)
            out_path = output_dir / f"{img_path.stem}_{name}_{pred_name}.png"
            Image.fromarray(viz).save(out_path)
            results.append({
                "image": img_path.name,
                "method": name,
                "predicted_class": pred_name,
                "confidence": float(probs[pred]),
                "saved": str(out_path),
            })
        print(f"{img_path.name}: pred={pred_name} (p={probs[pred]:.3f})")

    return results


def compute_heatmap_roi_iou(
    heatmap: np.ndarray,
    roi_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute IoU between a Grad-CAM heatmap and a binary ROI mask.

    Args:
        heatmap: H x W float array in [0, 1] (Grad-CAM output).
        roi_mask: H x W binary array (1 = pathologist-annotated ROI).
        threshold: Threshold to binarise the heatmap relative to its max.

    Returns:
        IoU in [0, 1].
    """
    if heatmap.shape != roi_mask.shape:
        raise ValueError(
            f"Shape mismatch: heatmap {heatmap.shape} vs ROI {roi_mask.shape}"
        )
    binary_heatmap = (heatmap >= threshold * heatmap.max()).astype(np.uint8)
    roi_mask = (roi_mask > 0).astype(np.uint8)
    intersection = (binary_heatmap & roi_mask).sum()
    union = (binary_heatmap | roi_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--model", default="modified_convnext")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from src.models.builder import build_model
    model = build_model(args.model, num_classes=4, pretrained=False)
    state = torch.load(args.weights, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    image_paths = sorted([p for p in args.images.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        print(f"No images found in {args.images}")
        return

    generate_heatmaps(
        model, image_paths, args.output_dir,
        device=args.device,
    )
    print(f"Heatmaps saved to {args.output_dir}")


if __name__ == "__main__":
    main()
