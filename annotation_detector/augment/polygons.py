"""Deriving YOLO polygons from masks, and checking they are usable."""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.004,
                    max_points: int = 60) -> np.ndarray | None:
    """Derive one normalised (N, 2) polygon from a binary mask.

    The largest external contour is used. YOLO segmentation is a single closed
    ring and cannot express holes, so a hollow mask such as a box border
    becomes a filled polygon along its outer edge.
    """
    binary = mask.astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        approx = contour

    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]

    h, w = mask.shape[:2]
    points[:, 0] /= float(w)
    points[:, 1] /= float(h)
    return np.clip(points, 0.0, 1.0)


def polygon_is_valid(polygon: np.ndarray | None,
                     min_area_norm: float = 2e-5) -> tuple[bool, str]:
    """Check one normalised polygon, returning (valid, reason)."""
    if polygon is None:
        return False, "empty polygon"
    if len(polygon) < 3:
        return False, f"fewer than 3 points ({len(polygon)})"
    if not np.isfinite(polygon).all():
        return False, "contains NaN/inf"
    if polygon.min() < -1e-6 or polygon.max() > 1.0 + 1e-6:
        return False, "outside [0,1]"

    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if area < min_area_norm:
        return False, f"area too small ({area:.2e})"
    return True, "ok"


def polygon_bbox_xyxy(polygon: np.ndarray, w: int, h: int) -> list[float]:
    """Pixel bounding box of a normalised polygon."""
    return [round(float(polygon[:, 0].min() * w), 2),
            round(float(polygon[:, 1].min() * h), 2),
            round(float(polygon[:, 0].max() * w), 2),
            round(float(polygon[:, 1].max() * h), 2)]
