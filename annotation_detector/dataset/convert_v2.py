"""V2 conversion to YOLO detection format with a stratified split."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ..constants import ANNOTATION_TYPES, CLASS_TO_ID


def xyxy_to_yolo(box, img_w: int, img_h: int):
    """Convert a pixel [x0, y0, x1, y1] box to normalised (cx, cy, w, h)."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h,
            (x1 - x0) / img_w, (y1 - y0) / img_h)


def load_annotations(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "annotations.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the V2 generator first.")
    return json.loads(path.read_text(encoding="utf-8"))


def split_records(records: list[dict], train_ratio: float, val_ratio: float,
                  seed: int = 42) -> dict:
    """Split stratified by (class, colour).

    Plain random splitting can leave a class absent from a split entirely,
    which makes that class's AP meaningless.
    """
    rng = random.Random(seed)

    groups: dict = {}
    for record in records:
        key = (record.get("annotation_type"), record.get("color"))
        groups.setdefault(key, []).append(record)

    out: dict = {"train": [], "val": [], "test": []}
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        bucket = groups[key][:]
        rng.shuffle(bucket)

        n = len(bucket)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))

        out["train"] += bucket[:n_train]
        out["val"] += bucket[n_train:n_train + n_val]
        out["test"] += bucket[n_train + n_val:]

    for split in out:
        rng.shuffle(out[split])
    return out


def make_yolo_dataset(dataset_dir: Path, output_dir: Path, train_ratio: float,
                      val_ratio: float, seed: int = 42) -> dict:
    records = load_annotations(dataset_dir)
    images_src = dataset_dir / "images"

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    splits = split_records(records, train_ratio, val_ratio, seed)
    active = {name: recs for name, recs in splits.items() if recs}

    for split in active:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    counts: dict = {}

    for split, recs in active.items():
        counts[split] = 0
        for record in recs:
            image_name = record["image"]
            src = images_src / image_name
            class_id = CLASS_TO_ID.get(record["annotation_type"])
            if not src.exists() or class_id is None:
                skipped += 1
                continue

            shutil.copy2(src, output_dir / "images" / split / image_name)

            img_w, img_h = record["image_width"], record["image_height"]
            lines = []
            for box in record["bounding_boxes_xyxy"]:
                cx, cy, w, h = xyxy_to_yolo(box, img_w, img_h)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path = (output_dir / "labels" / split /
                          (Path(image_name).stem + ".txt"))
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] += 1

    lines = [f"path: {output_dir.resolve()}"]
    lines += [f"{split}: images/{split}" for split in ("train", "val", "test")
              if split in active]
    lines.append(f"nc: {len(ANNOTATION_TYPES)}")
    lines.append(f"names: {ANNOTATION_TYPES}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")

    return {"counts": counts, "skipped": skipped, "active": active,
            "records": records}
