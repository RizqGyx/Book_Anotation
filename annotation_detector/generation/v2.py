"""V2 generator: clean synthetic pages, one mark each."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pymupdf as fitz

from ..constants import (ANNOTATION_TYPES, MARGIN, PAGE_HEIGHT, PAGE_WIDTH,
                         ZOOM)
from ..palette import COLOR_NAMES_V2, PALETTE_V2, color_for
from ..text import (CHUNK_MAX_WORDS_V2, chunk_cycler, chunk_sentences,
                    split_sentences)

FONT_SIZE = 11
FONT = "helv"
LINE_HEIGHT = 1.6
NOISE_PROBABILITY = 0.35
MAX_ATTEMPTS_MULTIPLIER = 6


def pick_target_sentence(chunk_sents: list[str], add_noise: bool = True) -> str:
    """Pick a sentence to annotate, sometimes bleeding into a neighbour.

    The bleed imitates a reader whose highlighter overshoots the sentence.
    """
    index = random.randint(0, len(chunk_sents) - 1)
    target = chunk_sents[index]

    if add_noise and random.random() < NOISE_PROBABILITY:
        if random.random() < 0.5 and index > 0:
            previous = chunk_sents[index - 1].split()
            n = random.randint(1, min(4, len(previous)))
            target = " ".join(previous[-n:]) + " " + target
        elif index < len(chunk_sents) - 1:
            following = chunk_sents[index + 1].split()
            n = random.randint(1, min(4, len(following)))
            target = target + " " + " ".join(following[:n])

    return target.strip()


def make_page(text_chunk: str):
    """Create a single-page in-memory PDF holding one text chunk."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    rect = fitz.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
    page.insert_textbox(rect, text_chunk, fontsize=FONT_SIZE, fontname=FONT,
                        align=0, lineheight=LINE_HEIGHT)
    return doc, page


def add_annotation(page, target_text: str, ann_type: str, rgb):
    """Mark target_text on the page and return its rects, or None."""
    quads = page.search_for(target_text, quads=True)
    if not quads:
        return None

    if ann_type == "highlight":
        annot = page.add_highlight_annot(quads)
    elif ann_type == "underline":
        annot = page.add_underline_annot(quads)
    elif ann_type == "squiggly":
        annot = page.add_squiggly_annot(quads)
    elif ann_type == "box":
        union = quads[0].rect
        for quad in quads[1:]:
            union |= quad.rect
        pad = 1.5
        union = fitz.Rect(union.x0 - pad, union.y0 - pad,
                          union.x1 + pad, union.y1 + pad)
        page.draw_rect(union, color=rgb, width=1.4)
        return [union]
    else:
        raise ValueError(f"Unknown annotation type: {ann_type}")

    annot.set_colors(stroke=rgb)
    annot.update()
    return [quad.rect for quad in quads]


def scale_bbox(rect, zoom: float = ZOOM) -> list[float]:
    """Convert a PDF-point rect into pixel coordinates of the rendered PNG."""
    return [round(rect.x0 * zoom, 2), round(rect.y0 * zoom, 2),
            round(rect.x1 * zoom, 2), round(rect.y1 * zoom, 2)]


def generate_dataset(source_text_path: str, output_dir: Path,
                     samples_per_combo: int = 50, color_names=None,
                     seed: int | None = 42) -> list[dict]:
    """Build a dataset balanced across every (class, colour) combination."""
    color_names = list(color_names or COLOR_NAMES_V2)
    unknown = [c for c in color_names if c not in PALETTE_V2]
    if unknown:
        raise ValueError(f"Unknown colours: {unknown}. "
                         f"Choose from: {COLOR_NAMES_V2}")

    rng = random.Random(seed)
    random.seed(seed)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    raw_text = Path(source_text_path).read_text(encoding="utf-8")
    sentences = split_sentences(raw_text)
    if len(sentences) < 10:
        raise ValueError("Source text is too short to build a dataset from.")

    chunks = chunk_sentences(sentences, CHUNK_MAX_WORDS_V2)
    combos = [(t, c) for t in ANNOTATION_TYPES for c in color_names]
    target_total = len(combos) * samples_per_combo

    print(f"Source     : {source_text_path}")
    print(f"Chunks     : {len(chunks)} text pages available")
    print(f"Classes    : {ANNOTATION_TYPES}")
    print(f"Colours    : {color_names}")
    print(f"Plan       : {len(combos)} combinations x {samples_per_combo} = "
          f"{target_total} images\n")

    cycler = chunk_cycler(chunks, rng)
    seen: set = set()
    dataset: list[dict] = []
    sample_id = 0

    for ann_type, color_name in combos:
        rgb = color_for(ann_type, color_name, PALETTE_V2)
        generated = attempts = 0
        max_attempts = samples_per_combo * MAX_ATTEMPTS_MULTIPLIER

        while generated < samples_per_combo and attempts < max_attempts:
            attempts += 1
            chunk_idx, chunk = next(cycler)
            chunk_sents = split_sentences(chunk)
            if len(chunk_sents) < 2:
                continue

            target_text = pick_target_sentence(chunk_sents)
            key = (chunk_idx, target_text)
            if key in seen:
                continue

            doc, page = make_page(chunk)
            bboxes = add_annotation(page, target_text, ann_type, rgb)
            if not bboxes:
                doc.close()
                continue

            image_name = f"{ann_type}_{color_name}_{sample_id:05d}.png"
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
            pix.save(str(images_dir / image_name))
            width, height = pix.width, pix.height
            doc.close()
            seen.add(key)

            dataset.append({
                "image": image_name,
                "image_width": width,
                "image_height": height,
                "annotation_type": ann_type,
                "color": color_name,
                "color_rgb": list(rgb),
                "annotated_text": target_text,
                "bounding_boxes_xyxy": [scale_bbox(b) for b in bboxes],
                "full_page_text": chunk,
            })

            generated += 1
            sample_id += 1

        flag = "" if generated == samples_per_combo else "  <-- SHORT"
        print(f"[{ann_type:<9} / {color_name:<6}] "
              f"{generated}/{samples_per_combo} samples "
              f"({attempts} attempts){flag}")

    (output_dir / "annotations.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone - {len(dataset)}/{target_total} samples")
    print("\nClass x colour matrix:")
    print(f"{'':<11}" + "".join(f"{c:>8}" for c in color_names))
    for ann_type in ANNOTATION_TYPES:
        row = f"{ann_type:<11}"
        for color_name in color_names:
            n = sum(1 for r in dataset
                    if r["annotation_type"] == ann_type
                    and r["color"] == color_name)
            row += f"{n:>8}"
        print(row)

    print(f"\nImages       : {images_dir}/")
    print(f"Ground truth : {output_dir / 'annotations.json'}")
    return dataset
