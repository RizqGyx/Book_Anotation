"""Converting V2 detection labels into the polygons YOLO segmentation needs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..constants import IMAGE_EXTS, SPLITS

LOGGER = logging.getLogger("annotation_detector.dataset.labels")


def bbox_line_to_polygon(parts: list[str]) -> str:
    """Turn `cls cx cy w h` into `cls x1 y1 x2 y2 x3 y3 x4 y4`.

    Exact rather than approximate: the annotation regions genuinely are
    axis-aligned rectangles, because they come from PyMuPDF text quads.
    """
    class_id = int(float(parts[0]))
    cx, cy, w, h = (float(v) for v in parts[1:5])
    x1 = max(0.0, min(1.0, cx - w / 2.0))
    y1 = max(0.0, min(1.0, cy - h / 2.0))
    x2 = max(0.0, min(1.0, cx + w / 2.0))
    y2 = max(0.0, min(1.0, cy + h / 2.0))
    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon)


def convert_label_file(src: Path, dst: Path) -> tuple[int, int]:
    """Write the polygon version of one label file; polygons pass through."""
    converted = passthrough = 0
    lines: list[str] = []

    for raw in src.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) == 5:
            lines.append(bbox_line_to_polygon(parts))
            converted += 1
        elif len(parts) >= 7 and len(parts) % 2 == 1:
            lines.append(raw.strip())
            passthrough += 1
        else:
            LOGGER.warning("Skipping unrecognised label line in %s: %r",
                           src.name, raw)

    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return converted, passthrough


def load_class_names(data_root: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def prepare_segmentation_dataset(data_root: Path,
                                 out_dir: Path) -> tuple[Path, list[str], dict]:
    """Build a polygon-labelled copy of a dataset, leaving the source alone."""
    names = load_class_names(data_root)
    LOGGER.info("Classes (%d): %s", len(names), names)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    stats: dict = {"classes": names, "splits": {}}
    available: list[str] = []

    for split in SPLITS:
        img_src = data_root / "images" / split
        lbl_src = data_root / "labels" / split
        if not img_src.is_dir():
            LOGGER.warning("Split '%s' missing, skipped.", split)
            continue

        img_dst = out_dir / "images" / split
        lbl_dst = out_dir / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_images = n_converted = n_passthrough = 0
        per_class = {name: 0 for name in names}

        for img_path in sorted(img_src.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_src / (img_path.stem + ".txt")
            if not lbl_path.exists():
                LOGGER.warning("No label for %s, image skipped.", img_path.name)
                continue

            shutil.copy2(img_path, img_dst / img_path.name)
            converted, passthrough = convert_label_file(lbl_path,
                                                        lbl_dst / lbl_path.name)
            n_images += 1
            n_converted += converted
            n_passthrough += passthrough

            written = (lbl_dst / lbl_path.name).read_text(encoding="utf-8")
            for line in written.splitlines():
                if line.strip():
                    index = int(float(line.split()[0]))
                    if 0 <= index < len(names):
                        per_class[names[index]] += 1

        stats["splits"][split] = {
            "images": n_images,
            "instances": n_converted + n_passthrough,
            "converted_from_bbox": n_converted,
            "already_polygon": n_passthrough,
            "instances_per_class": per_class,
        }
        available.append(split)
        LOGGER.info("  %-5s : %3d images, %3d instances (%d converted) | %s",
                    split, n_images, n_converted + n_passthrough, n_converted,
                    per_class)

    yaml_path = out_dir / "data.yaml"
    lines = [f"path: {out_dir.resolve()}"]
    lines += [f"{split}: images/{split}" for split in available]
    lines.append(f"nc: {len(names)}")
    lines.append(f"names: {names}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    LOGGER.info("Segmentation data.yaml written to: %s", yaml_path)
    stats["available_splits"] = available
    return yaml_path, names, stats
