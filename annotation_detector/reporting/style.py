"""Shared matplotlib styling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

C_TRAIN = "#2f6fdb"
C_VAL = "#e0642f"
C_TEST = "#2ca05a"
SPLIT_COLORS = [C_TRAIN, C_VAL, C_TEST]
