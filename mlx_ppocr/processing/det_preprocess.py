"""Detection preprocessing: resize, normalize, pad."""

import numpy as np
from PIL import Image


def det_preprocess(
    image: np.ndarray | Image.Image,
    limit_side_len: int = 960,
) -> tuple[np.ndarray, dict]:
    """Preprocess image for detection.

    Args:
        image: Input image (RGB PIL Image or HWC numpy array)
        limit_side_len: Max side length

    Returns:
        (preprocessed_array [1, H, W, 3], metadata dict with original shape info)
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    src_h, src_w = image.shape[:2]

    # Resize: longest side → limit_side_len, keep aspect ratio
    ratio = 1.0
    if max(src_h, src_w) > limit_side_len:
        ratio = limit_side_len / max(src_h, src_w)

    # Round down to multiple of 32 for compatible feature map sizes
    resize_h = max(32, int(src_h * ratio) // 32 * 32)
    resize_w = max(32, int(src_w * ratio) // 32 * 32)

    pil_img = Image.fromarray(image)
    if resize_h != src_h or resize_w != src_w:
        pil_img = pil_img.resize((resize_w, resize_h), Image.BILINEAR)
    image = np.array(pil_img)

    # Normalize: RGB order (standard ImageNet)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # Add batch dimension: [1, H, W, 3]
    image = np.expand_dims(image, 0)

    meta = {
        "src_h": src_h,
        "src_w": src_w,
        "resize_h": resize_h,
        "resize_w": resize_w,
        "ratio": ratio,
    }
    return image, meta
