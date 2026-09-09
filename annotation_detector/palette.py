"""Colour palettes for reader marks.

Every class draws from the same list of hues in equal numbers, so colour
carries no information about class. See docs/DESIGN_NOTES.md section 1.
"""

from __future__ import annotations

PALETTE_V2 = {
    "red":    {"highlight": (1.00, 0.55, 0.55), "pen": (0.85, 0.10, 0.10)},
    "orange": {"highlight": (1.00, 0.75, 0.45), "pen": (0.90, 0.45, 0.05)},
    "yellow": {"highlight": (1.00, 1.00, 0.40), "pen": (0.75, 0.62, 0.00)},
    "green":  {"highlight": (0.60, 1.00, 0.60), "pen": (0.10, 0.60, 0.20)},
    "blue":   {"highlight": (0.60, 0.80, 1.00), "pen": (0.10, 0.30, 0.85)},
    "purple": {"highlight": (0.78, 0.65, 1.00), "pen": (0.45, 0.15, 0.75)},
    "pink":   {"highlight": (1.00, 0.65, 0.85), "pen": (0.90, 0.20, 0.55)},
    "cyan":   {"highlight": (0.55, 0.95, 0.95), "pen": (0.00, 0.55, 0.65)},
}

PALETTE_V3 = {
    name: {**tones, "dark": False} for name, tones in PALETTE_V2.items()
}
PALETTE_V3.update({
    "dark_blue":   {"highlight": (0.62, 0.66, 0.80), "pen": (0.10, 0.14, 0.38),
                    "dark": True},
    "dark_gray":   {"highlight": (0.66, 0.66, 0.68), "pen": (0.24, 0.24, 0.26),
                    "dark": True},
    "deep_purple": {"highlight": (0.66, 0.60, 0.74), "pen": (0.24, 0.10, 0.34),
                    "dark": True},
    "dark_green":  {"highlight": (0.62, 0.72, 0.62), "pen": (0.08, 0.26, 0.12),
                    "dark": True},
})

COLOR_NAMES_V2 = list(PALETTE_V2)
COLOR_NAMES_V3 = list(PALETTE_V3)
DARK_COLORS = [c for c, v in PALETTE_V3.items() if v["dark"]]


def color_for(ann_type: str, color_name: str,
              palette: dict | None = None) -> tuple[float, float, float]:
    """Light tone for highlights, saturated tone for pen strokes.

    Highlight annotations multiply-blend, so a dark tone would bury the text.
    Pen strokes are thin and need a saturated tone to stay visible; pure yellow
    is darkened to gold for exactly that reason.
    """
    palette = palette if palette is not None else PALETTE_V3
    tone = "highlight" if ann_type == "highlight" else "pen"
    return palette[color_name][tone]


def is_dark(color_name: str, palette: dict | None = None) -> bool:
    palette = palette if palette is not None else PALETTE_V3
    return bool(palette.get(color_name, {}).get("dark", False))
