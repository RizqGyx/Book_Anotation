"""Lighting, shadow, glare and sensor effects. Masks are never touched here."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import AugConfig, LIGHTING_NAMES, LIGHTING_PROFILES


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_lighting(img: np.ndarray, profile: str, brightness: float,
                   contrast: float, saturation: float) -> np.ndarray:
    """Colour temperature, brightness, contrast and saturation."""
    r, g, b = LIGHTING_PROFILES.get(profile, LIGHTING_PROFILES["neutral"])
    out = img.astype(np.float32) * np.array([r, g, b], dtype=np.float32)

    out = out * brightness
    mean = out.mean()
    out = (out - mean) * contrast + mean

    if abs(saturation - 1.0) > 1e-3:
        gray = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = gray[..., None] + (out - gray[..., None]) * saturation

    return _u8(out)


def _shadow_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """One soft shadow shape: a hand, a phone, a page edge or an object."""
    h, w = shape
    shade = np.zeros((h, w), dtype=np.float32)
    kind = rng.choice(["hand", "phone", "page_edge", "object"])

    if kind == "hand":
        cx = int(rng.uniform(0.05, 0.95) * w)
        cy = int(rng.uniform(0.05, 0.95) * h)
        angle = float(rng.uniform(0, 180))
        for k in range(int(rng.integers(2, 5))):
            offset = int((k - 1.5) * w * 0.055)
            cv2.ellipse(shade, (cx + offset, cy),
                        (int(w * 0.055), int(h * 0.30)), angle, 0, 360, 1.0, -1)

    elif kind == "phone":
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        rect = ((cx, cy),
                (w * rng.uniform(0.25, 0.55), h * rng.uniform(0.20, 0.50)),
                float(rng.uniform(0, 180)))
        cv2.fillPoly(shade, [np.int32(cv2.boxPoints(rect))], 1.0)

    elif kind == "page_edge":
        side = rng.integers(0, 4)
        n = w if side in (0, 2) else h
        grad = np.linspace(1.0, 0.0, n, dtype=np.float32) ** 1.5
        shade = (np.tile(grad, (h, 1)) if side in (0, 2)
                 else np.tile(grad[:, None], (1, w)))
        if side == 3:
            shade = shade[::-1]
        elif side == 2:
            shade = shade[:, ::-1]

    else:
        n = int(rng.integers(3, 7))
        cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
        radius = rng.uniform(0.18, 0.45) * min(w, h)
        points = []
        for i in range(n):
            angle = 2 * math.pi * i / n + float(rng.uniform(-0.35, 0.35))
            rr = radius * float(rng.uniform(0.6, 1.35))
            points.append([cx + rr * math.cos(angle), cy + rr * math.sin(angle)])
        cv2.fillPoly(shade, [np.int32(points)], 1.0)

    kernel = int(max(w, h) * rng.uniform(0.03, 0.09)) | 1
    return np.clip(cv2.GaussianBlur(shade, (kernel, kernel), 0), 0.0, 1.0)


def apply_shadow(img: np.ndarray, rng: np.random.Generator,
                 strength: float) -> np.ndarray:
    """Multiply-blend a blurred shadow shape onto the page."""
    shade = _shadow_mask(img.shape[:2], rng) * strength
    return _u8(img.astype(np.float32) * (1.0 - shade[..., None]))


def apply_reflection(img: np.ndarray, rng: np.random.Generator,
                     strength: float) -> np.ndarray:
    """Screen-blend a specular streak or highlight, as on glossy paper."""
    h, w = img.shape[:2]
    glare = np.zeros((h, w), dtype=np.float32)

    if rng.random() < 0.6:
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        rect = ((cx, cy),
                (w * rng.uniform(0.35, 0.95), h * rng.uniform(0.03, 0.11)),
                float(rng.uniform(-70, 70)))
        cv2.fillPoly(glare, [np.int32(cv2.boxPoints(rect))], 1.0)
    else:
        centre = (int(rng.uniform(0.15, 0.85) * w),
                  int(rng.uniform(0.15, 0.85) * h))
        axes = (int(w * rng.uniform(0.10, 0.30)),
                int(h * rng.uniform(0.06, 0.20)))
        cv2.ellipse(glare, centre, axes, float(rng.uniform(0, 180)),
                    0, 360, 1.0, -1)

    kernel = int(max(w, h) * rng.uniform(0.04, 0.10)) | 1
    glare = np.clip(cv2.GaussianBlur(glare, (kernel, kernel), 0), 0.0, 1.0)
    glare = glare * strength

    base = img.astype(np.float32)
    return _u8(base + (255.0 - base) * glare[..., None])


def apply_sensor(img: np.ndarray, rng: np.random.Generator,
                 cfg: AugConfig) -> tuple[np.ndarray, dict]:
    """Focus blur and sensor noise, the fingerprints of a phone camera."""
    meta = {"noise_sigma": 0.0, "blur_sigma": 0.0}
    out = img

    if rng.random() < cfg.p_blur:
        sigma = float(rng.uniform(*cfg.blur_sigma))
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        meta["blur_sigma"] = round(sigma, 3)

    if rng.random() < cfg.p_noise:
        sigma = float(rng.uniform(*cfg.noise_sigma))
        noise = rng.normal(0, sigma, out.shape).astype(np.float32)
        out = _u8(out.astype(np.float32) + noise)
        meta["noise_sigma"] = round(sigma, 3)

    return out, meta


def apply_photometric(img: np.ndarray, cfg: AugConfig,
                      rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Run the whole photometric chain and report what was applied."""
    meta = {"lighting_type": "none", "brightness_factor": 1.0,
            "contrast_factor": 1.0, "saturation_factor": 1.0,
            "has_shadow": False, "shadow_strength": 0.0,
            "has_reflection": False, "reflection_strength": 0.0}
    out = img

    if rng.random() < cfg.p_lighting:
        profile = str(rng.choice(LIGHTING_NAMES))
        brightness = float(rng.uniform(*cfg.brightness))
        contrast = float(rng.uniform(*cfg.contrast))
        saturation = float(rng.uniform(*cfg.saturation))
        out = apply_lighting(out, profile, brightness, contrast, saturation)
        meta.update(lighting_type=profile,
                    brightness_factor=round(brightness, 3),
                    contrast_factor=round(contrast, 3),
                    saturation_factor=round(saturation, 3))

    if rng.random() < cfg.p_shadow:
        strength = float(rng.uniform(*cfg.shadow_strength))
        out = apply_shadow(out, rng, strength)
        meta.update(has_shadow=True, shadow_strength=round(strength, 3))

    if rng.random() < cfg.p_reflection:
        strength = float(rng.uniform(*cfg.reflection_strength))
        out = apply_reflection(out, rng, strength)
        meta.update(has_reflection=True, reflection_strength=round(strength, 3))

    out, sensor_meta = apply_sensor(out, rng, cfg)
    meta.update(sensor_meta)
    return out, meta
