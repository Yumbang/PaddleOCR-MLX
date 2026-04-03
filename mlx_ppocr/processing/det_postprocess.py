"""DB postprocessing: probability map → text bounding boxes."""

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def det_postprocess(
    prob_map: np.ndarray,
    src_h: int,
    src_w: int,
    resize_h: int,
    resize_w: int,
    thresh: float = 0.3,
    box_thresh: float = 0.6,
    max_candidates: int = 1000,
    unclip_ratio: float = 1.5,
) -> list[dict]:
    """Convert probability map to text bounding boxes.

    Args:
        prob_map: [H, W] probability map from detection model
        src_h, src_w: Original image dimensions
        resize_h, resize_w: Resized image dimensions
        thresh: Binarization threshold
        box_thresh: Minimum box confidence
        max_candidates: Max number of candidates
        unclip_ratio: Polygon expansion ratio

    Returns:
        List of dicts with 'points' (4x2 array) and 'score' (float)
    """
    # Binarize
    bitmap = (prob_map > thresh).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours[:max_candidates]:
        if contour.shape[0] < 4:
            continue

        # Score: mean probability inside contour
        score = _box_score(prob_map, contour.squeeze(1))
        if score < box_thresh:
            continue

        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect)

        # Unclip polygon (expand)
        points = _unclip(points, unclip_ratio)
        if points is None:
            continue

        # Get new minimum area rectangle after unclip
        rect = cv2.minAreaRect(points)
        points = cv2.boxPoints(rect)
        points = np.intp(points)

        # Scale back to original image coordinates
        points = points.astype(np.float32)
        points[:, 0] = points[:, 0] * src_w / resize_w
        points[:, 1] = points[:, 1] * src_h / resize_h
        points = np.clip(points, 0, [src_w, src_h]).astype(np.int32)

        # Sort points: top-left, top-right, bottom-right, bottom-left
        points = _order_points(points)

        results.append({"points": points, "score": float(score)})

    return results


def _box_score(prob_map: np.ndarray, points: np.ndarray) -> float:
    h, w = prob_map.shape
    xmin = max(0, int(points[:, 0].min()))
    xmax = min(w, int(points[:, 0].max()) + 1)
    ymin = max(0, int(points[:, 1].min()))
    ymax = min(h, int(points[:, 1].max()) + 1)
    if xmax <= xmin or ymax <= ymin:
        return 0.0

    mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
    pts = points.copy()
    pts[:, 0] -= xmin
    pts[:, 1] -= ymin
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return float(prob_map[ymin:ymax, xmin:xmax][mask.astype(bool)].mean())


def _unclip(points: np.ndarray, unclip_ratio: float) -> np.ndarray | None:
    poly = Polygon(points)
    if poly.area < 1:
        return None
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        [tuple(p) for p in points.astype(np.int64)],
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0], dtype=np.float32)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=pts.dtype)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect
