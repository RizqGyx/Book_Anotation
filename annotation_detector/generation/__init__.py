"""Rendering synthetic book pages and their reader annotations."""

from .config import MarkingConfig
from .page import add_margin_note, draw_marking, make_page, render_page_array
from .masks import region_masks_from_quads, true_masks_from_diff
from .placement import place_marking, select_span
from .sample import SampleResult, build_sample
from .report import build_generation_report, print_report

__all__ = [
    "MarkingConfig", "add_margin_note", "draw_marking", "make_page",
    "render_page_array", "region_masks_from_quads", "true_masks_from_diff",
    "place_marking", "select_span", "SampleResult", "build_sample",
    "build_generation_report", "print_report",
]
