"""Geometric transforms applied identically to image and masks."""

from __future__ import annotations

import cv2
import numpy as np

from .config import AugConfig


def masks_area(masks: list[np.ndarray]) -> np.ndarray:
    return np.array([int(m.sum()) for m in masks], dtype=np.int64)


def _warp_masks(masks: list[np.ndarray], fn) -> list[np.ndarray]:
    """Apply one warp to every mask, re-thresholding after interpolation."""
    return [fn(m.astype(np.uint8) * 255) > 127 for m in masks]


def surface_fill(rng: np.random.Generator) -> tuple[int, int, int]:
    """Background colour for areas exposed by rotation or perspective.

    A photographed page always shows the surface around it, so a neutral desk
    tone teaches a realistic page boundary where white would not.
    """
    base = int(rng.integers(70, 165))
    warm = int(rng.integers(-14, 22))
    return (max(0, min(255, base + warm)), base, max(0, min(255, base - warm // 2)))


def apply_rotation(img: np.ndarray, masks: list[np.ndarray], deg: float,
                   fill: tuple[int, int, int]):
    """Rotate about the image centre, keeping the canvas size."""
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)

    img_out = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    masks_out = _warp_masks(masks, lambda m: cv2.warpAffine(
        m, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return img_out, masks_out


def apply_perspective(img: np.ndarray, masks: list[np.ndarray], strength: float,
                      rng: np.random.Generator, fill: tuple[int, int, int]):
    """Homography with randomly displaced corners: a page shot at an angle."""
    h, w = img.shape[:2]
    dx, dy = strength * w, strength * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    dst = src + jitter * np.float32([dx, dy])
    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    img_out = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    masks_out = _warp_masks(masks, lambda m: cv2.warpPerspective(
        m, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return img_out, masks_out


def apply_page_curve(img: np.ndarray, masks: list[np.ndarray], amplitude: float,
                     rng: np.random.Generator, fill: tuple[int, int, int]):
    """Experimental page curvature near the binding, via cv2.remap.

    remap is a pure pixel mapping, so masks pass through exactly the same map
    and stay correct despite the transform being non-affine.
    """
    h, w = img.shape[:2]
    amp = amplitude * w
    side = 1.0 if rng.random() < 0.5 else -1.0

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    t = xx / max(w - 1, 1)
    weight = ((1.0 - t) if side > 0 else t) ** 1.6
    shift = amp * weight * np.sin(np.pi * yy / max(h - 1, 1)).astype(np.float32)

    map_x = (xx + side * shift).astype(np.float32)
    map_y = yy.astype(np.float32)

    def remap(src, border_value):
        return cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    return remap(img, fill), _warp_masks(masks, lambda m: remap(m, 0))


def _retention_ok(before: np.ndarray, after: list[np.ndarray],
                  min_retention: float) -> bool:
    """Reject transforms that push too much annotation out of frame."""
    if len(after) == 0:
        return True
    now = masks_area(after)
    if (now == 0).any():
        return False
    return bool((now / np.maximum(before, 1) >= min_retention).all())


def apply_geometric(img, masks, cfg: AugConfig, rng: np.random.Generator) -> tuple:
    """Run the geometric chain, weakening parameters until masks survive.

    If no attempt keeps enough of every mask, the untransformed image is
    returned: one less varied sample beats one wrong label.
    """
    meta = {"rotation_deg": 0.0, "perspective_strength": 0.0,
            "curve_amplitude": 0.0, "has_page_curve": False,
            "geom_attempts": 0, "geom_fallback": False}

    want_rotation = rng.random() < cfg.p_rotation
    want_perspective = rng.random() < cfg.p_perspective
    want_curve = rng.random() < cfg.p_page_curve
    if not (want_rotation or want_perspective or want_curve):
        return img, masks, meta

    area_before = masks_area(masks)
    fill = surface_fill(rng)

    for attempt in range(cfg.max_geom_attempts):
        damp = 1.0 - attempt / (cfg.max_geom_attempts + 1)
        image, current = img, masks
        applied = {"rotation_deg": 0.0, "perspective_strength": 0.0,
                   "curve_amplitude": 0.0, "has_page_curve": False}

        if want_curve:
            amount = float(rng.uniform(*cfg.curve_amplitude)) * damp
            image, current = apply_page_curve(image, current, amount, rng, fill)
            applied["curve_amplitude"] = round(amount, 5)
            applied["has_page_curve"] = True

        if want_perspective:
            amount = float(rng.uniform(*cfg.perspective_strength)) * damp
            image, current = apply_perspective(image, current, amount, rng, fill)
            applied["perspective_strength"] = round(amount, 5)

        if want_rotation:
            degrees = float(rng.uniform(*cfg.rotation_deg)) * damp
            degrees *= 1 if rng.random() < 0.5 else -1
            image, current = apply_rotation(image, current, degrees, fill)
            applied["rotation_deg"] = round(degrees, 3)

        if _retention_ok(area_before, current, cfg.min_mask_retention):
            meta.update(applied, geom_attempts=attempt + 1, geom_fallback=False)
            return image, current, meta

    meta["geom_attempts"] = cfg.max_geom_attempts
    meta["geom_fallback"] = True
    return img, masks, meta
