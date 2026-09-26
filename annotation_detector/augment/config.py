"""Augmentation probabilities, parameter ranges and lighting profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AugConfig:
    """Probabilities and parameter ranges for every augmentation."""

    p_rotation: float = 0.50
    rotation_deg: tuple[float, float] = (2.0, 12.0)

    p_perspective: float = 0.40
    perspective_strength: tuple[float, float] = (0.010, 0.055)

    p_page_curve: float = 0.15
    curve_amplitude: tuple[float, float] = (0.008, 0.030)

    p_lighting: float = 0.60
    brightness: tuple[float, float] = (0.82, 1.15)
    contrast: tuple[float, float] = (0.90, 1.12)
    saturation: tuple[float, float] = (0.85, 1.15)

    p_shadow: float = 0.35
    shadow_strength: tuple[float, float] = (0.20, 0.55)

    p_reflection: float = 0.20
    reflection_strength: tuple[float, float] = (0.15, 0.42)

    p_noise: float = 0.35
    noise_sigma: tuple[float, float] = (1.5, 6.0)

    p_blur: float = 0.20
    blur_sigma: tuple[float, float] = (0.4, 1.1)

    min_mask_retention: float = 0.60
    max_geom_attempts: int = 6

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "light": dict(p_rotation=0.30, p_perspective=0.20, p_lighting=0.45,
                  p_shadow=0.18, p_reflection=0.10, p_page_curve=0.05,
                  p_noise=0.20, p_blur=0.10, rotation_deg=(1.0, 6.0),
                  perspective_strength=(0.008, 0.030)),
    "default": dict(),
    "heavy": dict(p_rotation=0.70, p_perspective=0.60, p_lighting=0.75,
                  p_shadow=0.50, p_reflection=0.35, p_page_curve=0.25,
                  p_noise=0.50, p_blur=0.30, rotation_deg=(3.0, 14.0),
                  perspective_strength=(0.015, 0.070)),
}

LIGHTING_PROFILES = {
    "neutral":     (1.000, 1.000, 1.000),
    "warm_lamp":   (1.085, 1.005, 0.890),
    "warm_sunset": (1.120, 0.985, 0.855),
    "cool_shade":  (0.905, 0.975, 1.095),
    "cool_led":    (0.940, 0.995, 1.070),
    "daylight":    (1.020, 1.000, 0.985),
}
LIGHTING_NAMES = list(LIGHTING_PROFILES)
