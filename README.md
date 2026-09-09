# Reader Annotation Detection

Instance segmentation for the four ways people mark up a physical book page.

| id | class | what it looks like |
|----|-------|--------------------|
| 0 | `highlight` | a filled band behind the text |
| 1 | `underline` | a thin straight line under a line of text |
| 2 | `squiggly` | a wavy line under a line of text |
| 3 | `box` | a rectangle drawn around a passage, hollow inside |

Model: **`yolo26n-seg.pt`** (YOLO26 nano, instance segmentation, 2.7 M parameters).

Training data is **fully synthetic**. Pages are rendered from a public-domain
Project Gutenberg text with PyMuPDF, and the marks are drawn programmatically,
so ground truth comes from the layout engine rather than from a human
annotator. No manual labelling is involved anywhere in this repository.

---

## Why there are two datasets

The interesting part of this project is not the training loop. It is what the
measurements exposed about the data.

**V2** produces flawless pages: one font, one size, upright, evenly lit, one
annotation each. A model reaches **0.995 mask mAP@0.50** on it. That number
looks like success and is mostly an illusion — there is nothing left to learn
and no evidence the model would survive a real photograph.

**V3** keeps V2 untouched as a clean baseline and adds the variation a phone
camera actually produces: tilt, perspective, warm and cool lighting, hand
shadows, glare on glossy paper, nine fonts, thin and faded strokes, several
marks per page, and pages with no annotation at all.

The cross evaluation is the point:

| model on test set | mask mAP@0.50-0.95 |
|---|---|
| V2 on V2 (clean) | 0.766 |
| **V2 on V3 (realistic)** | **0.110** |
| **V3 on V3 (realistic)** | **0.797** |
| V3 on V2 (clean) | 0.497 |

The clean model loses **86%** of its accuracy the moment a page is tilted,
shadowed, or set in a different typeface. The V3 model scores *higher* on the
hard test set than V2 manages on its own easy one, and it produced **zero false
positives** across 27 blank pages.

The cross matrix exists so this cannot be misread. A lower V3 score on the V2
test set could otherwise be taken as "worse model" when it is mostly "different
label distribution and a much harder training task" — and those are very
different conclusions.

---

## Quick start

```bash
# 1. environment (uv is a fast drop-in for pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. build a dataset
python scripts/generate_dataset.py data/book.txt --version v3
python scripts/convert_dataset.py --version v3

# 3. validate, train, evaluate, compare
python scripts/train.py --version v3 --epochs 60
```

Everything lands in `output_v3/`, starting with `REPORT.md`.

For the clean V2 baseline instead, pass `--version v2` to all three.

`ultralytics >= 8.4.0` is required: the YOLO26 checkpoints only exist from that
release onward. Device selection is automatic (CUDA, then Apple MPS, then CPU);
override with `--device`.

---

## Repository layout

```
annotation_detector/          the package
├── constants.py              classes, splits, IoU thresholds, page geometry
├── palette.py                colour palettes and the light/saturated tones
├── fonts.py                  font discovery with graceful fallback
├── text.py                   book text into page-sized chunks
├── augment/                  config, geometric, photometric, polygons
├── generation/               page, masks, placement, sample, v2, v3, report
├── dataset/                  labels, convert_v2, convert_v3, validate
├── evaluation/               detection, masks, overfitting, robustness,
│                             negatives, holdout, cross
├── reporting/                plots, plots_v3, tables, markdown
└── pipeline/                 common, v2, v3

scripts/                      command line entry points
├── generate_dataset.py       build a synthetic dataset
├── convert_dataset.py        convert it to YOLO format
├── validate_dataset.py       quality gate
└── train.py                  train, evaluate, report

data/book.txt                 source text (public domain)
real_holdout/                 drop real book photos here (never trained on)
docs/DESIGN_NOTES.md          why the non-obvious decisions are what they are
docs/V2_PIPELINE.md           V2 reference
docs/V3_PIPELINE.md           V3 reference
```

Every script works from a clone with no install step. `pip install -e .` is
supported if you would rather import the package from elsewhere.

Datasets, training runs and weights are gitignored — they total several
gigabytes and every one of them regenerates from a fixed seed.

---

## What the pipeline measures

Beyond overall mAP, which is where most projects stop:

**Full IoU sweep.** mAP at every threshold from 0.50 to 0.95 in steps of 0.05,
for the box branch and the mask branch, on train, val and test. A curve that
collapses between 0.50 and 0.95 means objects are found but their boundaries
are sloppy — invisible if you only report mAP@0.50.

**Mask-level metrics** computed by rasterising predictions and ground truth:
mean mask IoU, Dice, per-class pixel IoU, pixel accuracy, and instance
precision, recall and F1 at IoU 0.50.

**Overfitting, from two independent angles.** The loss curves (where val loss
bottomed out, how far it climbed afterwards, the linear trend across the last
30% of epochs) and the generalisation gap (the model re-evaluated on the train
split and compared against val and test). Combined into a 0-8 score and a
verdict.

**Robustness by condition** (V3 only). Generation metadata splits the test
results into groups — upright vs tilted, shadowed vs clean, serif vs sans,
thin vs thick strokes, partial vs full marks, small vs large type. The report
ranks conditions by F1 spread, so the weakest one is the first thing you see.

**False positives on blank pages** (V3 only). 15% of V3 pages carry no
annotation at all. A detector that flags something on every page floods the
user with phantom highlights, and no mAP number will tell you that is
happening.

---

## Design decisions worth knowing

Full reasoning lives in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md). The
three that matter most:

**Colour carries no class information.** Every class appears in every palette
colour exactly the same number of times. An earlier version gave highlights
pastel tones and pen strokes red ones, which let the model score well by
learning "red means not a highlight" without ever looking at the shape. The
palette is now shared across all four classes by construction.

**Labels are masks, not polygons.** Every geometric augmentation is applied
with the identical operator to the image and to each instance mask, and
polygons are re-derived from the transformed masks afterwards. The image and
its label cannot drift apart, and page curvature — a non-affine `cv2.remap` —
works correctly, which point-wise polygon transforms cannot handle.

**Splits are grouped, not just stratified.** Several V3 pages are rendered from
the same source text chunk. Splitting per image would put two variants of the
same page in train and test at once, and the test score would be optimistic for
the wrong reason. All variants of one chunk stay in one split, and the split
report verifies this explicitly.

---

## Known limitations

- **Still synthetic.** Rendered from PDF, not photographed. No paper texture,
  no real curvature, no camera JPEG artefacts. An approximation, not a
  substitute for real data.
- **One source text**, in English prose, single column, no tables or figures.
- **Rectangular masks** by default. True stroke masks are implemented behind
  `--true-mask` but are not the default; see `docs/DESIGN_NOTES.md` for the
  two independent reasons why.
- **`box` yields one instance** while the other classes yield one per line, so
  its instance count is roughly half theirs. Consistent with how people draw
  boxes, but it shows up in the per-class metrics.
- **Font availability is machine-dependent.** Nine fonts resolve on this macOS
  machine; use `--no-system-fonts` for reproducible cross-machine runs.
- **Never tested on real photographs yet.** `real_holdout/` is ready and empty.
- **The two test sets are not perfectly interchangeable.** V3 polygons come
  from `approxPolyDP` of augmented masks, V2 boxes from exact text rectangles,
  so the mask branch of the cross matrix carries a small label-convention
  difference. The box branch is the cleaner comparison.

---

## Using a trained model

```python
from ultralytics import YOLO

model = YOLO("output_v3/weights/best.pt")
result = model.predict("page.jpg", conf=0.25, retina_masks=True)[0]

for polygon, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
    print(result.names[int(cls)], float(conf), polygon.shape)
```

The default confidence of 0.25 was chosen for metrics, not for users. A single
phantom highlight is more annoying than a missed one, so a product would
probably want it higher — the precision-recall curves in `output_v3/plots/` are
there to choose it deliberately.
# Book_Anotation
