"""Extracting detection and mask mAP from an Ultralytics validation run."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..constants import IOU_THRESHOLDS

LOGGER = logging.getLogger("annotation_detector.evaluation")


def metric_branch_to_dict(metric, names: list[str]) -> dict:
    """Flatten one Ultralytics Metric branch, including the full IoU sweep."""
    all_ap = np.asarray(metric.all_ap, dtype=float)
    class_index = np.asarray(metric.ap_class_index, dtype=int).tolist()

    precision = np.asarray(metric.p, dtype=float)
    recall = np.asarray(metric.r, dtype=float)
    f1 = np.asarray(metric.f1, dtype=float)

    out: dict = {
        "precision": float(np.mean(precision)) if precision.size else 0.0,
        "recall": float(np.mean(recall)) if recall.size else 0.0,
        "f1": float(np.mean(f1)) if f1.size else 0.0,
        "mAP50": float(metric.map50),
        "mAP75": float(metric.map75),
        "mAP50-95": float(metric.map),
    }

    per_iou = (all_ap.mean(axis=0) if all_ap.size
               else np.zeros(len(IOU_THRESHOLDS)))
    out["mAP_per_iou"] = {f"{t:.2f}": float(v)
                          for t, v in zip(IOU_THRESHOLDS, per_iou)}
    out["mAP50-95_from_sweep"] = float(per_iou.mean())

    per_class = {}
    for row, index in enumerate(class_index):
        name = names[index] if 0 <= index < len(names) else str(index)
        per_class[name] = {
            "precision": float(precision[row]) if row < precision.size else 0.0,
            "recall": float(recall[row]) if row < recall.size else 0.0,
            "f1": float(f1[row]) if row < f1.size else 0.0,
            "AP50": float(all_ap[row, 0]),
            "AP75": float(all_ap[row, 5]),
            "AP50-95": float(all_ap[row].mean()),
            "AP_per_iou": {f"{t:.2f}": float(v)
                           for t, v in zip(IOU_THRESHOLDS, all_ap[row])},
        }
    out["per_class"] = per_class
    return out


def evaluate_split(model, data_yaml: Path, split: str, out_dir: Path,
                   names: list[str], imgsz: int, batch: int,
                   device: str) -> dict:
    """Run the Ultralytics validator on one split and summarise the result."""
    LOGGER.info("-- Evaluating split '%s' --", split)
    result = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                       batch=batch, device=device, project=str(out_dir),
                       name=f"val_{split}", exist_ok=True, plots=True,
                       verbose=True)

    summary = {
        "split": split,
        "box": metric_branch_to_dict(result.box, names),
        "mask": metric_branch_to_dict(result.seg, names),
        "speed_ms": {k: float(v) for k, v in dict(result.speed).items()},
        "fitness": float(result.fitness) if result.fitness is not None else None,
    }
    for branch, label in (("box", "Box "), ("mask", "Mask")):
        data = summary[branch]
        LOGGER.info("   %s | mAP50 %.4f  mAP75 %.4f  mAP50-95 %.4f  "
                    "P %.4f  R %.4f", label, data["mAP50"], data["mAP75"],
                    data["mAP50-95"], data["precision"], data["recall"])
    return summary
