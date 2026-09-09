"""How reader marks vary: thickness, fade, count, line span, partiality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarkingConfig:
    underline_width: tuple[float, float] = (0.8, 3.2)
    squiggly_width: tuple[float, float] = (0.7, 2.6)
    box_width: tuple[float, float] = (0.9, 3.0)
    highlight_pad: tuple[float, float] = (-0.8, 2.6)

    p_faded: float = 0.30
    faded_opacity: tuple[float, float] = (0.30, 0.62)
    normal_opacity: tuple[float, float] = (0.72, 1.00)
    highlight_opacity_scale: float = 0.80
    dark_highlight_opacity: tuple[float, float] = (0.28, 0.50)

    markings_per_page: tuple[int, ...] = (1, 2, 3, 4)
    markings_weights: tuple[float, ...] = (0.42, 0.33, 0.18, 0.07)

    line_targets: tuple[int, ...] = (1, 2, 3, 4)
    line_weights: tuple[float, ...] = (0.30, 0.35, 0.25, 0.10)

    p_partial: float = 0.45
    p_margin_note: float = 0.20
