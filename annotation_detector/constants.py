"""Values shared across generation, training and evaluation."""

from __future__ import annotations

import numpy as np

ANNOTATION_TYPES = ["highlight", "underline", "squiggly", "box"]
CLASS_TO_ID = {name: i for i, name in enumerate(ANNOTATION_TYPES)}
CLASS_NAMES = ANNOTATION_TYPES

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

IOU_THRESHOLDS = np.round(np.linspace(0.50, 0.95, 10), 2)
MASK_METRIC_CONF = 0.25
MASK_METRIC_IOU_MATCH = 0.50

PAGE_WIDTH = 420
PAGE_HEIGHT = 595
MARGIN = 40
ZOOM = 2.0

CLASS_COLORS = [
    (0.95, 0.77, 0.06),
    (0.20, 0.63, 0.94),
    (0.36, 0.78, 0.42),
    (0.90, 0.30, 0.35),
]

UNICODE_FALLBACKS = {
    "—": " - ", "–": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...", "′": "'", "″": '"',
    " ": " ", "﻿": "", "­": "",
}
