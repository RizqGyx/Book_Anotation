"""Building per-instance masks for a mark."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..constants import ZOOM

TRUE_MASK_DIFF_THRESHOLD = 12
MIN_COMPONENT_AREA = 40


def region_masks_from_quads(quads, ann_type: str, h: int, w: int,
                            zoom: float = ZOOM) -> list[np.ndarray]:
    """Masks covering the rectangular text region that was marked.

    One instance per line, except box which is a single merged instance
    because a reader draws one box, not one per line.
    """
    out: list[np.ndarray] = []

    def rect_mask(rect) -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        x0, y0 = max(0, int(rect.x0 * zoom)), max(0, int(rect.y0 * zoom))
        x1 = min(w, int(math.ceil(rect.x1 * zoom)))
        y1 = min(h, int(math.ceil(rect.y1 * zoom)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    if ann_type == "box":
        union = quads[0].rect
        for quad in quads[1:]:
            union |= quad.rect
        mask = rect_mask(union)
        if mask.any():
            out.append(mask)
    else:
        for quad in quads:
            mask = rect_mask(quad.rect)
            if mask.any():
                out.append(mask)
    return out


def true_masks_from_diff(img_before: np.ndarray, img_after: np.ndarray,
                         ann_type: str) -> list[np.ndarray]:
    """Experimental stroke masks from the pixel difference around a mark.

    Both renders are deterministic, so the difference is exactly the pixels the
    mark changed. See docs/DESIGN_NOTES.md section 5 for why this is not the
    default.
    """
    diff = np.abs(img_after.astype(np.int16) - img_before.astype(np.int16))
    mask = (diff.max(axis=2) > TRUE_MASK_DIFF_THRESHOLD).astype(np.uint8)
    if mask.sum() == 0:
        return []

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels = cv2.connectedComponents(mask)
    out = [labels == i for i in range(1, count)
           if (labels == i).sum() >= MIN_COMPONENT_AREA]

    if ann_type == "box" and len(out) > 1:
        merged = np.zeros_like(out[0])
        for component in out:
            merged |= component
        return [merged]
    return out
