"""Overfitting judged from the loss curves and from the generalisation gap."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..constants import SPLITS

LOGGER = logging.getLogger("annotation_detector.evaluation")


def normalised_slope(values: np.ndarray) -> float:
    """Linear-fit slope divided by mean |y|, so scales stay comparable."""
    if values.size < 3:
        return 0.0
    slope = float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])
    return slope / (float(np.mean(np.abs(values))) or 1.0)


def _generalisation_gap(eval_results: dict) -> dict:
    gaps = {}
    for branch in ("box", "mask"):
        train, val, test = (
            eval_results.get(split, {}).get(branch, {}).get("mAP50-95")
            for split in SPLITS)
        gaps[branch] = {
            "train_mAP50-95": train, "val_mAP50-95": val, "test_mAP50-95": test,
            "gap_train_minus_val": (train - val)
            if (train is not None and val is not None) else None,
            "gap_train_minus_test": (train - test)
            if (train is not None and test is not None) else None,
            "relative_gap_val_pct": ((train - val) / train * 100.0)
            if (train and val is not None) else None,
        }
    return gaps


def _score_signals(rise_pct: float, val_slope: float, train_slope: float,
                   best_val_epoch: int, gaps: dict, map_curve: np.ndarray,
                   best_map_index: int, n_epochs: int) -> tuple[int, list[str]]:
    signals: list[str] = []
    score = 0

    if rise_pct >= 5.0 and val_slope > 0.001:
        score += 2
        signals.append(f"val loss rose {rise_pct:.1f}% above its minimum "
                       f"(epoch {best_val_epoch}) and is still trending up")
    elif rise_pct >= 2.0:
        score += 1
        signals.append(f"val loss rose slightly ({rise_pct:.1f}%) after epoch "
                       f"{best_val_epoch}")

    if train_slope < -0.001 and val_slope > 0.0:
        score += 2
        signals.append("train loss still falling while val loss climbs")

    relative_gap = gaps["mask"]["relative_gap_val_pct"]
    if relative_gap is not None:
        if relative_gap >= 30.0:
            score += 2
            signals.append(f"train mask mAP50-95 is {relative_gap:.1f}% above val")
        elif relative_gap >= 15.0:
            score += 1
            signals.append("moderate train/val mask mAP50-95 gap "
                           f"({relative_gap:.1f}%)")

    if (best_map_index >= 0
            and n_epochs - best_map_index > max(10, 0.4 * n_epochs)
            and map_curve[-1] < map_curve[best_map_index] * 0.95):
        score += 1
        signals.append(f"mAP peaked at epoch {best_map_index + 1} "
                       "then fell over 5%")

    return score, signals


def analyze_overfitting(results_csv: Path, eval_results: dict) -> dict:
    """A val loss turning upward while train loss falls is the classic sign;
    the mAP gap between splits is the independent second view."""
    import pandas as pd

    frame = pd.read_csv(results_csv)
    frame.columns = [c.strip() for c in frame.columns]

    train_cols = [c for c in frame.columns
                  if c.startswith("train/") and c.endswith("loss")]
    val_cols = [c for c in frame.columns
                if c.startswith("val/") and c.endswith("loss")]
    LOGGER.info("Train loss columns : %s", train_cols)
    LOGGER.info("Val loss columns   : %s", val_cols)

    train_total = frame[train_cols].sum(axis=1).to_numpy(dtype=float)
    val_total = frame[val_cols].sum(axis=1).to_numpy(dtype=float)
    epochs = (frame["epoch"].to_numpy(dtype=float) if "epoch" in frame.columns
              else np.arange(1, len(frame) + 1))
    n_epochs = len(frame)

    best_val_index = int(np.argmin(val_total))
    min_val = float(val_total[best_val_index])
    final_val = float(val_total[-1])
    final_train = float(train_total[-1])
    rise_pct = ((final_val - min_val) / min_val * 100.0) if min_val > 0 else 0.0

    tail = max(3, int(round(n_epochs * 0.30)))
    val_slope = normalised_slope(val_total[-tail:])
    train_slope = normalised_slope(train_total[-tail:])

    map_col = next((c for c in frame.columns
                    if "mAP50-95" in c and "(M)" in c), None)
    map_col = map_col or next((c for c in frame.columns if "mAP50-95" in c), None)
    map_curve = (frame[map_col].to_numpy(dtype=float) if map_col
                 else np.array([]))
    best_map_index = int(np.argmax(map_curve)) if map_curve.size else -1

    gaps = _generalisation_gap(eval_results)
    score, signals = _score_signals(rise_pct, val_slope, train_slope,
                                    best_val_index + 1, gaps, map_curve,
                                    best_map_index, n_epochs)

    if score >= 4:
        verdict = "OVERFITTING"
        advice = ("Use best.pt rather than last.pt, add data or augmentation, "
                  "enable early stopping (--patience 20-30), or train fewer "
                  "epochs.")
    elif score >= 2:
        verdict = "MILD OVERFITTING"
        advice = ("Still under control. best.pt already comes from the best "
                  "epoch; consider more data if the gap widens.")
    else:
        verdict = "NO SIGNIFICANT OVERFITTING"
        advice = ("Train and val move together. If both losses are still "
                  "falling steeply at the end, the model is underfitting - "
                  "train longer.")
        if train_slope < -0.005 and val_slope < -0.005:
            signals.append("both losses still falling at the end - likely underfit")

    if not signals:
        signals.append("no overfitting signal in the curves or the gap")

    analysis = {
        "epochs_run": int(n_epochs),
        "loss_columns": {"train": train_cols, "val": val_cols},
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "min_val_loss": min_val,
        "best_val_loss_epoch": (int(epochs[best_val_index]) if len(epochs)
                                else best_val_index + 1),
        "val_loss_rise_after_min_pct": rise_pct,
        "val_loss_trend_last_30pct_per_epoch": val_slope,
        "train_loss_trend_last_30pct_per_epoch": train_slope,
        "val_over_train_loss_ratio": ((final_val / final_train)
                                      if final_train else None),
        "map_column": map_col,
        "best_map_epoch": (int(epochs[best_map_index])
                           if best_map_index >= 0 else None),
        "best_map_value": (float(map_curve[best_map_index])
                           if best_map_index >= 0 else None),
        "final_map_value": float(map_curve[-1]) if map_curve.size else None,
        "generalization_gap": gaps,
        "overfitting_score": score,
        "verdict": verdict,
        "signals": signals,
        "advice": advice,
        "_curves": {
            "epochs": epochs, "train_total": train_total,
            "val_total": val_total, "df": frame,
            "train_cols": train_cols, "val_cols": val_cols,
        },
    }

    LOGGER.info("VERDICT: %s (score %d/8)", verdict, score)
    for signal in signals:
        LOGGER.info("  - %s", signal)
    LOGGER.info("  Advice: %s", advice)
    return analysis
