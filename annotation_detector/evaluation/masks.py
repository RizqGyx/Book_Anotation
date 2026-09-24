"""Mask-level metrics Ultralytics does not report directly."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..constants import IMAGE_EXTS, MASK_METRIC_CONF, MASK_METRIC_IOU_MATCH

LOGGER = logging.getLogger("annotation_detector.evaluation")


def read_polygons(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """Read a polygon label file into (class_id, pixel points) pairs."""
    polygons: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        points = coords.reshape(-1, 2) * np.array([w, h], dtype=np.float32)
        polygons.append((class_id, points))
    return polygons


def rasterize(points: np.ndarray, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def iou_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    sum_a, sum_b = a.sum(dtype=np.int64), b.sum(dtype=np.int64)
    union = sum_a + sum_b - intersection
    return (float(intersection / union) if union else 0.0,
            float(2 * intersection / (sum_a + sum_b)) if (sum_a + sum_b) else 0.0)


def _predict_masks(model, image_path: Path, imgsz: int, device: str,
                   conf: float):
    result = model.predict(source=str(image_path), imgsz=imgsz, conf=conf,
                           device=device, retina_masks=True, verbose=False)[0]
    h, w = result.orig_shape
    predictions: list[tuple[int, float, np.ndarray]] = []
    if result.masks is not None and len(result.masks) > 0:
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for polygon, confidence, class_id in zip(result.masks.xy, confidences,
                                                 classes):
            if len(polygon) >= 3:
                predictions.append((
                    int(class_id), float(confidence),
                    rasterize(np.asarray(polygon, dtype=np.float32), w, h)))
    predictions.sort(key=lambda item: item[1], reverse=True)
    return predictions, w, h


def _safe_div(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_mask_metrics(model, seg_root: Path, split: str, names: list[str],
                         imgsz: int, device: str) -> dict:
    """Mean mask IoU and Dice over matched pairs, instance P/R/F1 at IoU 0.50,
    and dataset-wide pixel IoU, Dice and accuracy."""
    LOGGER.info("-- Mask metrics for '%s' (conf>=%.2f, match IoU>=%.2f) --",
                split, MASK_METRIC_CONF, MASK_METRIC_IOU_MATCH)

    image_dir = seg_root / "images" / split
    label_dir = seg_root / "labels" / split
    image_paths = sorted(p for p in image_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)

    n_classes = len(names)
    matched_iou: list[float] = []
    matched_dice: list[float] = []
    tp = fp = fn = 0
    per_class = {name: {"iou": [], "dice": [], "tp": 0, "fp": 0, "fn": 0}
                 for name in names}
    pixel_intersection = np.zeros(n_classes, dtype=np.int64)
    pixel_union = np.zeros(n_classes, dtype=np.int64)
    pixel_gt = np.zeros(n_classes, dtype=np.int64)
    pixel_pred = np.zeros(n_classes, dtype=np.int64)
    correct_pixels = total_pixels = 0

    for image_path in image_paths:
        predictions, w, h = _predict_masks(model, image_path, imgsz, device,
                                           MASK_METRIC_CONF)
        ground_truth = [
            (class_id, rasterize(points, w, h)) for class_id, points
            in read_polygons(label_dir / (image_path.stem + ".txt"), w, h)]

        used = [False] * len(ground_truth)
        for class_id, _confidence, prediction in predictions:
            best_index, best_iou, best_dice = -1, 0.0, 0.0
            for index, (gt_class, gt_mask) in enumerate(ground_truth):
                if used[index] or gt_class != class_id:
                    continue
                iou, dice = iou_dice(prediction, gt_mask)
                if iou > best_iou:
                    best_index, best_iou, best_dice = index, iou, dice
            name = names[class_id] if 0 <= class_id < n_classes else str(class_id)
            if best_index >= 0 and best_iou >= MASK_METRIC_IOU_MATCH:
                used[best_index] = True
                tp += 1
                matched_iou.append(best_iou)
                matched_dice.append(best_dice)
                per_class[name]["tp"] += 1
                per_class[name]["iou"].append(best_iou)
                per_class[name]["dice"].append(best_dice)
            else:
                fp += 1
                per_class[name]["fp"] += 1

        for index, (gt_class, _mask) in enumerate(ground_truth):
            if not used[index]:
                fn += 1
                name = names[gt_class] if 0 <= gt_class < n_classes else str(gt_class)
                per_class[name]["fn"] += 1

        gt_any = np.zeros((h, w), dtype=bool)
        pred_any = np.zeros((h, w), dtype=bool)
        for class_id in range(n_classes):
            gt_mask = np.zeros((h, w), dtype=bool)
            for gt_class, mask in ground_truth:
                if gt_class == class_id:
                    gt_mask |= mask
            pred_mask = np.zeros((h, w), dtype=bool)
            for pred_class, _confidence, mask in predictions:
                if pred_class == class_id:
                    pred_mask |= mask
            pixel_intersection[class_id] += np.logical_and(
                gt_mask, pred_mask).sum(dtype=np.int64)
            pixel_union[class_id] += np.logical_or(
                gt_mask, pred_mask).sum(dtype=np.int64)
            pixel_gt[class_id] += gt_mask.sum(dtype=np.int64)
            pixel_pred[class_id] += pred_mask.sum(dtype=np.int64)
            gt_any |= gt_mask
            pred_any |= pred_mask
        correct_pixels += int((gt_any == pred_any).sum())
        total_pixels += gt_any.size

    per_class_out = {}
    for class_id, name in enumerate(names):
        data = per_class[name]
        per_class_out[name] = {
            "instances_gt": data["tp"] + data["fn"],
            "TP": data["tp"], "FP": data["fp"], "FN": data["fn"],
            "precision@IoU0.50": _safe_div(data["tp"], data["tp"] + data["fp"]),
            "recall@IoU0.50": _safe_div(data["tp"], data["tp"] + data["fn"]),
            "mean_mask_IoU_matched": (float(np.mean(data["iou"]))
                                      if data["iou"] else 0.0),
            "mean_Dice_matched": (float(np.mean(data["dice"]))
                                  if data["dice"] else 0.0),
            "pixel_IoU": _safe_div(pixel_intersection[class_id],
                                   pixel_union[class_id]),
            "pixel_Dice": _safe_div(2 * pixel_intersection[class_id],
                                    pixel_gt[class_id] + pixel_pred[class_id]),
            "pixel_precision": _safe_div(pixel_intersection[class_id],
                                         pixel_pred[class_id]),
            "pixel_recall": _safe_div(pixel_intersection[class_id],
                                      pixel_gt[class_id]),
        }

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    out = {
        "split": split,
        "conf_threshold": MASK_METRIC_CONF,
        "match_iou_threshold": MASK_METRIC_IOU_MATCH,
        "images": len(image_paths),
        "TP": tp, "FP": fp, "FN": fn,
        "instance_precision@IoU0.50": precision,
        "instance_recall@IoU0.50": recall,
        "instance_F1@IoU0.50": _safe_div(2 * precision * recall,
                                         precision + recall),
        "mean_mask_IoU_matched": (float(np.mean(matched_iou))
                                  if matched_iou else 0.0),
        "median_mask_IoU_matched": (float(np.median(matched_iou))
                                    if matched_iou else 0.0),
        "mean_Dice_matched": (float(np.mean(matched_dice))
                              if matched_dice else 0.0),
        "mean_pixel_IoU_over_classes": float(
            np.mean([per_class_out[n]["pixel_IoU"] for n in names])),
        "mean_pixel_Dice_over_classes": float(
            np.mean([per_class_out[n]["pixel_Dice"] for n in names])),
        "pixel_accuracy": _safe_div(correct_pixels, total_pixels),
        "per_class": per_class_out,
        "matched_iou_values": [round(v, 4) for v in matched_iou],
    }
    LOGGER.info("   mask mIoU %.4f | Dice %.4f | pixel IoU %.4f | "
                "P %.4f R %.4f F1 %.4f | TP %d FP %d FN %d",
                out["mean_mask_IoU_matched"], out["mean_Dice_matched"],
                out["mean_pixel_IoU_over_classes"], precision, recall,
                out["instance_F1@IoU0.50"], tp, fp, fn)
    return out
