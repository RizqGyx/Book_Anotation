"""Plots shared by the V2 and V3 pipelines."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..constants import CLASS_COLORS, IOU_THRESHOLDS, MASK_METRIC_CONF, SPLITS
from ..evaluation.masks import read_polygons
from .style import C_TEST, C_TRAIN, C_VAL, SPLIT_COLORS, plt


def plot_loss_components(analysis: dict, out: Path) -> None:
    curves = analysis["_curves"]
    frame, epochs = curves["df"], curves["epochs"]
    pairs = []
    for train_col in curves["train_cols"]:
        val_col = "val/" + train_col.split("/", 1)[1]
        if val_col in frame.columns:
            title = train_col.split("/", 1)[1].replace("_loss", "")
            pairs.append((title, train_col, val_col))
    if not pairs:
        return

    ncol = min(len(pairs), 4)
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow),
                             squeeze=False)
    for ax, (title, train_col, val_col) in zip(axes.ravel(), pairs):
        ax.plot(epochs, frame[train_col], color=C_TRAIN, lw=1.6, label="train")
        ax.plot(epochs, frame[val_col], color=C_VAL, lw=1.6, label="val")
        ax.set_title(f"{title} loss")
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Loss per component - train vs validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "01_loss_components.png")
    plt.close(fig)


def plot_overfitting(analysis: dict, out: Path) -> None:
    curves = analysis["_curves"]
    epochs = curves["epochs"]
    train, val = curves["train_total"], curves["val_total"]
    best = int(np.argmin(val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(epochs, train, color=C_TRAIN, lw=1.8, label="total train loss")
    ax1.plot(epochs, val, color=C_VAL, lw=1.8, label="total val loss")
    ax1.axvline(epochs[best], color="#888", ls="--", lw=1.2)
    ax1.annotate(f"val loss minimum\nepoch {int(epochs[best])}",
                 xy=(epochs[best], val[best]), xytext=(0.42, 0.82),
                 textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="#666"),
                 fontsize=8, color="#444")
    if best < len(epochs) - 1:
        ax1.axvspan(epochs[best], epochs[-1], color=C_VAL, alpha=0.07)
        ax1.text(0.985, 0.05, "overfitting risk zone", transform=ax1.transAxes,
                 ha="right", fontsize=8, color=C_VAL)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("total loss")
    ax1.set_title("Total loss: train vs validation")
    ax1.legend(frameon=False)

    gap = val - train
    ax2.plot(epochs, gap, color="#7b4bd8", lw=1.8)
    ax2.axhline(0, color="#999", lw=1)
    ax2.fill_between(epochs, 0, gap, where=(gap > 0), color="#7b4bd8", alpha=0.15)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val loss - train loss")
    ax2.set_title("Generalisation gap per epoch")

    verdict = analysis["verdict"]
    color = {"OVERFITTING": "#c0392b",
             "MILD OVERFITTING": "#d68910"}.get(verdict, "#1e8449")
    fig.suptitle(f"Overfitting diagnosis - {verdict} "
                 f"(score {analysis['overfitting_score']}/8)",
                 fontweight="bold", color=color)
    fig.tight_layout()
    fig.savefig(out / "02_overfitting_diagnosis.png")
    plt.close(fig)


def plot_metric_curves(analysis: dict, out: Path) -> None:
    curves = analysis["_curves"]
    frame, epochs = curves["df"], curves["epochs"]
    columns = [c for c in frame.columns if c.startswith("metrics/")]
    if not columns:
        return
    groups = {"mAP50": [], "mAP50-95": [], "precision": [], "recall": []}
    for column in columns:
        for key in groups:
            if key in column and not (key == "mAP50" and "mAP50-95" in column):
                groups[key].append(column)
    groups = {k: v for k, v in groups.items() if v}

    fig, axes = plt.subplots(1, len(groups), figsize=(4.0 * len(groups), 3.4),
                             squeeze=False)
    for ax, (key, cols) in zip(axes.ravel(), groups.items()):
        for i, column in enumerate(cols):
            label = ("Mask (M)" if "(M)" in column
                     else ("Box (B)" if "(B)" in column else column))
            ax.plot(epochs, frame[column], lw=1.6,
                    color=[C_TRAIN, C_TEST][i % 2], label=label)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
    fig.suptitle("Validation metrics across training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "03_metric_curves.png")
    plt.close(fig)


def plot_iou_sweep(eval_results: dict, out: Path) -> None:
    """mAP at every IoU threshold, with the 0.50-0.95 mean as a dashed line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = dict(zip(SPLITS, SPLIT_COLORS))

    for ax, branch, title in zip(axes, ("box", "mask"),
                                 ("Box (bounding box)", "Mask (segmentation)")):
        for split in SPLITS:
            data = eval_results.get(split, {}).get(branch)
            if not data:
                continue
            values = [data["mAP_per_iou"][f"{t:.2f}"] for t in IOU_THRESHOLDS]
            ax.plot(IOU_THRESHOLDS, values, "o-", lw=1.8, ms=4.5,
                    color=colors[split],
                    label=f"{split} (mAP50-95 = {data['mAP50-95']:.3f})")
            ax.axhline(data["mAP50-95"], color=colors[split], ls=":", lw=1.1,
                       alpha=0.7)
        ax.set_xlabel("IoU threshold")
        ax.set_title(title)
        ax.set_xticks(IOU_THRESHOLDS)
        ax.set_xticklabels([f"{t:.2f}" for t in IOU_THRESHOLDS], rotation=45)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("mAP")
    fig.suptitle("IoU sweep: mAP@0.50 through mAP@0.95", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "04_iou_sweep.png")
    plt.close(fig)


def plot_per_class_ap(eval_results: dict, names: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(names))
    for ax, split in zip(axes, ("val", "test")):
        data = eval_results.get(split, {}).get("mask", {}).get("per_class", {})
        ap50 = [data.get(n, {}).get("AP50", 0.0) for n in names]
        ap = [data.get(n, {}).get("AP50-95", 0.0) for n in names]
        ax.bar(x - 0.2, ap50, 0.4, label="AP@0.50", color="#5b8ff9")
        ax.bar(x + 0.2, ap, 0.4, label="AP@0.50-0.95", color="#f6a54a")
        for i, (a, b) in enumerate(zip(ap50, ap)):
            ax.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"split: {split}")
        ax.legend(frameon=False)
    axes[0].set_ylabel("AP (mask)")
    fig.suptitle("Per-class mask AP", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "05_per_class_ap.png")
    plt.close(fig)


def plot_generalization(eval_results: dict, out: Path) -> None:
    keys = ["mAP50", "mAP75", "mAP50-95", "precision", "recall"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(keys))
    width = 0.26
    for ax, branch, title in zip(axes, ("box", "mask"), ("Box (B)", "Mask (M)")):
        for i, split in enumerate(SPLITS):
            data = eval_results.get(split, {}).get(branch)
            if not data:
                continue
            values = [data.get(k, 0.0) for k in keys]
            ax.bar(x + (i - 1) * width, values, width, label=split,
                   color=SPLIT_COLORS[i])
            for xi, value in zip(x + (i - 1) * width, values):
                ax.text(xi, value + 0.015, f"{value:.2f}", ha="center",
                        fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0].set_ylabel("value")
    fig.suptitle("Generalisation gap: train vs val vs test", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "06_generalization_gap.png")
    plt.close(fig)


def plot_mask_quality(mask_metrics: dict, names: list[str], out: Path) -> None:
    splits = [s for s in SPLITS if s in mask_metrics]
    if not splits:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    keys = ["mean_mask_IoU_matched", "mean_Dice_matched",
            "mean_pixel_IoU_over_classes", "instance_precision@IoU0.50",
            "instance_recall@IoU0.50", "instance_F1@IoU0.50"]
    labels = ["mask IoU", "Dice", "pixel IoU", "P@0.5", "R@0.5", "F1@0.5"]
    x = np.arange(len(keys))
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        values = [mask_metrics[split].get(k, 0.0) for k in keys]
        axes[0].bar(x + (i - (len(splits) - 1) / 2) * width, values, width,
                    label=split, color=SPLIT_COLORS[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(frameon=False)
    axes[0].set_title("Mask quality summary")

    x = np.arange(len(names))
    for i, split in enumerate(splits):
        values = [mask_metrics[split]["per_class"].get(n, {}).get("pixel_IoU", 0.0)
                  for n in names]
        axes[1].bar(x + (i - (len(splits) - 1) / 2) * width, values, width,
                    label=split, color=SPLIT_COLORS[i])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(frameon=False)
    axes[1].set_title("Pixel IoU per class")

    data = [mask_metrics[s]["matched_iou_values"] or [0.0] for s in splits]
    try:
        box = axes[2].boxplot(data, tick_labels=splits, patch_artist=True,
                              widths=0.5)
    except TypeError:
        box = axes[2].boxplot(data, labels=splits, patch_artist=True, widths=0.5)
    for patch, color in zip(box["boxes"], SPLIT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    axes[2].axhline(0.5, color="#c0392b", ls="--", lw=1, label="match threshold")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Per-instance matched IoU")

    fig.suptitle("Mask and pixel level segmentation metrics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "07_mask_quality.png")
    plt.close(fig)


def plot_predictions(model, seg_root: Path, split: str, names: list[str],
                     imgsz: int, device: str, out: Path,
                     max_images: int = 6) -> None:
    """Side-by-side ground truth and prediction for a few images."""
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image

    image_dir = seg_root / "images" / split
    label_dir = seg_root / "labels" / split
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:max_images]
    if not paths:
        return

    fig, axes = plt.subplots(len(paths), 2, figsize=(7.2, 4.6 * len(paths)),
                             squeeze=False)
    for row, path in enumerate(paths):
        image = np.asarray(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]

        ax = axes[row][0]
        ax.imshow(image)
        for class_id, points in read_polygons(label_dir / (path.stem + ".txt"),
                                              w, h):
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            ax.add_patch(MplPolygon(points, closed=True, fill=True, alpha=0.30,
                                    facecolor=color, edgecolor=color, lw=1.4))
        ax.set_title(f"GT - {path.name}", fontsize=8)
        ax.axis("off")

        ax = axes[row][1]
        ax.imshow(image)
        result = model.predict(source=str(path), imgsz=imgsz,
                               conf=MASK_METRIC_CONF, device=device,
                               retina_masks=True, verbose=False)[0]
        if result.masks is not None:
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for polygon, confidence, class_id in zip(result.masks.xy,
                                                     confidences, classes):
                if len(polygon) < 3:
                    continue
                color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
                ax.add_patch(MplPolygon(np.asarray(polygon), closed=True,
                                        fill=True, alpha=0.30, facecolor=color,
                                        edgecolor=color, lw=1.4))
                ax.text(polygon[:, 0].min(), max(polygon[:, 1].min() - 6, 10),
                        f"{names[int(class_id)]} {confidence:.2f}", fontsize=6.5,
                        color="white",
                        bbox=dict(facecolor=color, edgecolor="none", pad=1.2))
        ax.set_title("Prediction", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Ground truth vs prediction - {split} split", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"08_predictions_{split}.png", bbox_inches="tight")
    plt.close(fig)


def copy_ultralytics_plots(train_dir: Path, out: Path) -> list[str]:
    """Copy Ultralytics' own plots so every output sits in one place."""
    copied = []
    for src in sorted(train_dir.glob("*.png")) + sorted(train_dir.glob("*.jpg")):
        dst = out / f"ultralytics_{src.name}"
        shutil.copy2(src, dst)
        copied.append(dst.name)
    return copied
