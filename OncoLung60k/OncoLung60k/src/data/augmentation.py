"""Augmentation transforms for training and evaluation."""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(
    image_size: int = 256,
    is_training: bool = True,
    color_jitter: float = 0.1,
    rotation_deg: float = 15.0,
    flip: bool = True,
):
    """Build a torchvision transform pipeline.

    Args:
        image_size: Output image side length.
        is_training: If True, includes augmentation.
        color_jitter: ColorJitter strength for brightness/contrast/saturation.
        rotation_deg: Random rotation range in degrees.
        flip: If True, includes horizontal/vertical flips.
    """
    if is_training:
        ops = [
            transforms.Resize((image_size, image_size)),
        ]
        if flip:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        if rotation_deg > 0:
            ops.append(transforms.RandomRotation(rotation_deg))
        if color_jitter > 0:
            ops.append(
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter / 2,
                )
            )
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    else:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

    return transforms.Compose(ops)
