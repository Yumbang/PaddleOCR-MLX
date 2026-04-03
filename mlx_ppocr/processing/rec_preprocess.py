"""Recognition preprocessing: crop text region and resize."""

import cv2
import numpy as np
from PIL import Image


def crop_text_region(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and perspective-correct a text region from image.

    Args:
        image: Full image (HWC, RGB)
        points: 4x2 polygon points (top-left, top-right, bottom-right, bottom-left)

    Returns:
        Cropped and rectified text region (HWC, RGB)
    """
    points = points.astype(np.float32)
    w = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[3] - points[2]),
    )
    h = max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
    )
    w, h = int(w), int(h)
    if w < 1 or h < 1:
        return None

    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points, dst_pts)
    crop = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # If the text is vertical (tall and narrow), rotate
    if h / w >= 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return crop


def rec_preprocess(
    crop: np.ndarray,
    target_height: int = 48,
    target_width: int = 320,
) -> np.ndarray:
    """Preprocess cropped text region for recognition.

    Args:
        crop: Cropped text image (HWC, RGB)
        target_height: Fixed height
        target_width: Max width (pad if shorter)

    Returns:
        Preprocessed array [1, H, W, 3]
    """
    return rec_preprocess_crop(crop, target_height, target_width)[np.newaxis, :]


def rec_preprocess_crop(
    crop: np.ndarray,
    target_height: int = 48,
    max_width: int = 320,
) -> np.ndarray:
    """Preprocess a single crop without adding batch dimension.

    Args:
        crop: Cropped text image (HWC, RGB)
        target_height: Fixed height
        max_width: Max width (pad if shorter)

    Returns:
        Preprocessed array [H, W, 3] padded to max_width
    """
    h, w = crop.shape[:2]

    # Resize height to target, keep aspect ratio
    ratio = target_height / h
    new_w = min(int(w * ratio), max_width)
    new_w = max(new_w, 1)

    pil_img = Image.fromarray(crop)
    pil_img = pil_img.resize((new_w, target_height), Image.BILINEAR)
    crop = np.array(pil_img)

    # Normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1.0
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5

    # Pad to max_width
    padded = np.zeros((target_height, max_width, 3), dtype=np.float32)
    padded[:, :new_w, :] = crop

    return padded
