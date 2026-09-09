"""Font discovery. Never assume a font exists on another machine."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

BUILTIN_FONTS = [
    {"alias": "helv", "name": "Helvetica", "family": "sans", "builtin": True},
    {"alias": "tiro", "name": "Times-Roman", "family": "serif", "builtin": True},
    {"alias": "cour", "name": "Courier", "family": "mono", "builtin": True},
]

CANDIDATE_SYSTEM_FONTS = [
    {"alias": "palatino", "name": "Palatino", "family": "serif",
     "path": "/System/Library/Fonts/Palatino.ttc"},
    {"alias": "newyork", "name": "New York", "family": "serif",
     "path": "/System/Library/Fonts/NewYork.ttf"},
    {"alias": "georgia", "name": "Georgia", "family": "serif",
     "path": "/Library/Fonts/Georgia.ttf"},
    {"alias": "helvneue", "name": "Helvetica Neue", "family": "sans",
     "path": "/System/Library/Fonts/HelveticaNeue.ttc"},
    {"alias": "avenir", "name": "Avenir", "family": "sans",
     "path": "/System/Library/Fonts/Avenir.ttc"},
    {"alias": "optima", "name": "Optima", "family": "sans",
     "path": "/System/Library/Fonts/Optima.ttc"},
    {"alias": "sfrounded", "name": "SF Rounded", "family": "sans",
     "path": "/System/Library/Fonts/SFNSRounded.ttf"},
    {"alias": "verdana", "name": "Verdana", "family": "sans",
     "path": "/Library/Fonts/Verdana.ttf"},
]

HANDWRITING_CANDIDATES = [
    {"alias": "chalk", "name": "Chalkboard",
     "path": "/System/Library/Fonts/Supplemental/Chalkboard.ttc"},
    {"alias": "noteworthy", "name": "Noteworthy",
     "path": "/System/Library/Fonts/Supplemental/Noteworthy.ttc"},
    {"alias": "marker", "name": "Marker Felt",
     "path": "/System/Library/Fonts/Supplemental/MarkerFelt.ttc"},
    {"alias": "bradley", "name": "Bradley Hand",
     "path": "/System/Library/Fonts/Supplemental/BradleyHandITCTT-Bold.ttf"},
]

FONT_SIZES = [
    {"name": "small", "size": 9.0, "lineheight": 1.55},
    {"name": "medium", "size": 11.0, "lineheight": 1.60},
    {"name": "large", "size": 13.0, "lineheight": 1.65},
]


def _try_load(candidate: dict, skipped: list[str], label: str = "") -> bool:
    path = Path(candidate["path"])
    prefix = f"{label}{candidate['name']}"
    if not path.exists():
        skipped.append(f"{prefix}: file not found ({path})")
        return False
    try:
        fitz.Font(fontfile=str(path))
        return True
    except Exception as exc:
        skipped.append(f"{prefix}: load failed ({type(exc).__name__})")
        return False


def discover_fonts(allow_system: bool = True):
    """Return (text_fonts, handwriting_fonts, skipped_notes)."""
    fonts = list(BUILTIN_FONTS)
    skipped: list[str] = []

    if allow_system:
        for candidate in CANDIDATE_SYSTEM_FONTS:
            if _try_load(candidate, skipped):
                fonts.append({**candidate, "builtin": False})

    handwriting: list[dict] = []
    for candidate in HANDWRITING_CANDIDATES:
        if _try_load(candidate, skipped, label="[handwriting] "):
            handwriting.append(candidate)

    return fonts, handwriting, skipped
