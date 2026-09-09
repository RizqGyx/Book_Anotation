"""Laying out a page and drawing marks on it."""

from __future__ import annotations

import math
import random

import numpy as np
import pymupdf as fitz

from ..constants import MARGIN, PAGE_HEIGHT, PAGE_WIDTH, ZOOM
from ..text import fit_text_to_page
from .config import MarkingConfig


def make_page(doc, text_chunk: str, font: dict, size_cfg: dict,
              fit_to_size: bool = True):
    """Lay one text chunk onto a fresh page in the chosen font and size."""
    if fit_to_size:
        text_chunk = fit_text_to_page(text_chunk, size_cfg["size"])
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    if not font.get("builtin", False):
        page.insert_font(fontname=font["alias"], fontfile=font["path"])
    rect = fitz.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
    page.insert_textbox(rect, text_chunk, fontsize=size_cfg["size"],
                        fontname=font["alias"], align=0,
                        lineheight=size_cfg["lineheight"])
    return page


def render_page_array(page, zoom: float = ZOOM) -> np.ndarray:
    """Render a page to an RGB uint8 array."""
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def _draw_highlight(page, quads, rgb, opacity: float, pad: float) -> None:
    for quad in quads:
        r = quad.rect
        band = fitz.Rect(r.x0 - 0.5, r.y0 - pad * 0.35,
                         r.x1 + 0.5, r.y1 + pad * 0.65)
        page.draw_rect(band, color=None, fill=rgb,
                       fill_opacity=opacity, overlay=False)


def _draw_underline(page, quads, rgb, opacity: float, thickness: float,
                    rng: random.Random) -> None:
    for quad in quads:
        r = quad.rect
        y = r.y1 - thickness * 0.35
        jitter = rng.uniform(-0.6, 0.6)
        page.draw_line(fitz.Point(r.x0, y + jitter), fitz.Point(r.x1, y - jitter),
                       color=rgb, width=thickness, stroke_opacity=opacity)


def _draw_squiggly(page, quads, rgb, opacity: float, thickness: float,
                   rng: random.Random) -> None:
    for quad in quads:
        r = quad.rect
        amp = rng.uniform(0.9, 1.9)
        period = rng.uniform(4.0, 7.5)
        y0 = r.y1 - amp
        n = max(6, int((r.x1 - r.x0) / period * 2))
        points = [fitz.Point(r.x0 + (r.x1 - r.x0) * i / n,
                             y0 + amp * math.sin(2 * math.pi * i / max(n / 2.2, 1)))
                  for i in range(n + 1)]
        page.draw_polyline(points, color=rgb, width=thickness,
                           stroke_opacity=opacity)


def _draw_box(page, quads, rgb, opacity: float, thickness: float,
              rng: random.Random) -> None:
    union = quads[0].rect
    for quad in quads[1:]:
        union |= quad.rect
    pad = rng.uniform(1.0, 3.5)
    rect = fitz.Rect(union.x0 - pad, union.y0 - pad,
                     union.x1 + pad, union.y1 + pad)
    page.draw_rect(rect, color=rgb, width=thickness, stroke_opacity=opacity)


def draw_marking(page, quads, ann_type: str, rgb, rng: random.Random,
                 mcfg: MarkingConfig, is_dark: bool) -> dict:
    """Draw one mark by hand and return its style metadata.

    Drawing manually rather than using PDF annotation objects is what allows
    stroke width and opacity to vary. Highlights use overlay=False so they sit
    beneath the text like a real highlighter.
    """
    faded = rng.random() < mcfg.p_faded
    low, high = mcfg.faded_opacity if faded else mcfg.normal_opacity
    opacity = rng.uniform(low, high)

    if ann_type == "highlight":
        if is_dark:
            opacity = rng.uniform(*mcfg.dark_highlight_opacity)
        else:
            opacity *= mcfg.highlight_opacity_scale
        thickness = rng.uniform(*mcfg.highlight_pad)
        _draw_highlight(page, quads, rgb, opacity, thickness)
        thickness = round(thickness, 2)

    elif ann_type == "underline":
        thickness = rng.uniform(*mcfg.underline_width)
        _draw_underline(page, quads, rgb, opacity, thickness, rng)

    elif ann_type == "squiggly":
        thickness = rng.uniform(*mcfg.squiggly_width)
        _draw_squiggly(page, quads, rgb, opacity, thickness, rng)

    elif ann_type == "box":
        thickness = rng.uniform(*mcfg.box_width)
        _draw_box(page, quads, rgb, opacity, thickness, rng)

    else:
        raise ValueError(f"Unknown class: {ann_type}")

    return {"marking_thickness": round(float(thickness), 3),
            "marking_opacity": round(float(opacity), 3),
            "is_faded": bool(faded)}


MARGIN_NOTES = ["important!", "check again", "compare", "NB", "?", "see ch. 3",
                "agreed", "good point", "unclear", "remember this"]


def add_margin_note(page, hand_fonts: list[dict], rng: random.Random) -> bool:
    """Add handwriting in the margin as a distractor, never as a label.

    It tests whether the detector mistakes handwriting for a reader mark.
    Falls back to synthetic scribbles when no handwriting font is installed.
    """
    ink = (rng.uniform(0.05, 0.35), rng.uniform(0.05, 0.35),
           rng.uniform(0.15, 0.5))
    left = rng.random() < 0.5
    x0 = 4 if left else PAGE_WIDTH - MARGIN + 3
    y0 = rng.uniform(MARGIN, PAGE_HEIGHT - MARGIN - 60)

    if hand_fonts and rng.random() < 0.65:
        font = rng.choice(hand_fonts)
        try:
            page.insert_font(fontname=font["alias"], fontfile=font["path"])
            page.insert_textbox(fitz.Rect(x0, y0, x0 + MARGIN - 6, y0 + 55),
                                rng.choice(MARGIN_NOTES),
                                fontsize=rng.uniform(6.5, 9.0),
                                fontname=font["alias"], color=ink, rotate=0)
            return True
        except Exception:
            pass

    for _ in range(rng.randint(2, 4)):
        y = y0 + rng.uniform(0, 42)
        n = rng.randint(8, 16)
        width = MARGIN - 10
        points = [fitz.Point(x0 + width * i / n, y + rng.uniform(-2.2, 2.2))
                  for i in range(n + 1)]
        page.draw_polyline(points, color=ink, width=rng.uniform(0.5, 1.1),
                           stroke_opacity=rng.uniform(0.55, 0.9))
    return True
