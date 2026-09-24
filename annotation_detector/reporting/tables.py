"""CSV exports and the Markdown table helper."""

from __future__ import annotations

from pathlib import Path

from ..constants import SPLITS


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return "-" if value is None else str(value)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def write_tables(eval_results: dict, mask_metrics: dict, names: list[str],
                 metrics_dir: Path) -> None:
    import pandas as pd

    rows = []
    for split in SPLITS:
        for branch in ("box", "mask"):
            data = eval_results.get(split, {}).get(branch)
            if not data:
                continue
            rows.append({
                "split": split, "branch": branch,
                "precision": data["precision"], "recall": data["recall"],
                "f1": data["f1"], "mAP@0.50": data["mAP50"],
                "mAP@0.75": data["mAP75"], "mAP@0.50-0.95": data["mAP50-95"],
            })
    pd.DataFrame(rows).to_csv(metrics_dir / "summary.csv", index=False)

    rows = []
    for split in SPLITS:
        for branch in ("box", "mask"):
            data = eval_results.get(split, {}).get(branch)
            if not data:
                continue
            row = {"split": split, "branch": branch}
            row.update({f"mAP@{k}": v for k, v in data["mAP_per_iou"].items()})
            row["mAP@0.50-0.95 (mean)"] = data["mAP50-95"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_dir / "iou_sweep.csv", index=False)

    rows = []
    for split in SPLITS:
        for branch in ("box", "mask"):
            per_class = eval_results.get(split, {}).get(branch, {}).get(
                "per_class", {})
            for name, data in per_class.items():
                rows.append({"split": split, "branch": branch, "class": name,
                             "precision": data["precision"],
                             "recall": data["recall"], "f1": data["f1"],
                             "AP@0.50": data["AP50"], "AP@0.75": data["AP75"],
                             "AP@0.50-0.95": data["AP50-95"]})
    pd.DataFrame(rows).to_csv(metrics_dir / "per_class.csv", index=False)

    rows = []
    for split, metrics in mask_metrics.items():
        rows.append({
            "split": split, "class": "ALL",
            "mean_mask_IoU": metrics["mean_mask_IoU_matched"],
            "mean_Dice": metrics["mean_Dice_matched"],
            "pixel_IoU": metrics["mean_pixel_IoU_over_classes"],
            "pixel_Dice": metrics["mean_pixel_Dice_over_classes"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "precision@IoU0.50": metrics["instance_precision@IoU0.50"],
            "recall@IoU0.50": metrics["instance_recall@IoU0.50"],
            "F1@IoU0.50": metrics["instance_F1@IoU0.50"],
            "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"],
        })
        for name in names:
            data = metrics["per_class"].get(name, {})
            rows.append({
                "split": split, "class": name,
                "mean_mask_IoU": data.get("mean_mask_IoU_matched"),
                "mean_Dice": data.get("mean_Dice_matched"),
                "pixel_IoU": data.get("pixel_IoU"),
                "pixel_Dice": data.get("pixel_Dice"),
                "pixel_accuracy": None,
                "precision@IoU0.50": data.get("precision@IoU0.50"),
                "recall@IoU0.50": data.get("recall@IoU0.50"),
                "F1@IoU0.50": None,
                "TP": data.get("TP"), "FP": data.get("FP"), "FN": data.get("FN"),
            })
    pd.DataFrame(rows).to_csv(metrics_dir / "mask_metrics.csv", index=False)
