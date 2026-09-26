"""Pieces both pipelines share: logging, device, seeds, model, training."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("annotation_detector")


def setup_logging(log_dir: Path) -> Path:
    """Log to stdout and to file, capturing Ultralytics output as well."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False

    ultralytics_logger = logging.getLogger("ultralytics")
    ultralytics_logger.setLevel(logging.INFO)
    ultralytics_logger.addHandler(file_handler)

    return log_file


def section(title: str) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info(title)
    LOGGER.info("=" * 78)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, fallback: str):
    """Load the requested checkpoint, falling back if it cannot be fetched."""
    from ultralytics import YOLO

    try:
        model = YOLO(model_name)
        LOGGER.info("Model loaded: %s", model_name)
        return model, model_name
    except Exception as exc:
        LOGGER.error("Could not load '%s': %s", model_name, exc)
        LOGGER.warning("Falling back to '%s'.", fallback)
        return YOLO(fallback), fallback


def train(model, data_yaml: Path, out_dir: Path, device: str,
          args) -> float:
    """Run training and return the elapsed minutes.

    patience 0 maps to more than the epoch count, which disables early stopping
    deliberately: the overfitting analysis needs the whole curve.
    """
    kwargs = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=device, workers=args.workers, seed=args.seed,
        deterministic=True,
        patience=args.patience if args.patience > 0 else args.epochs + 1,
        optimizer=args.optimizer, lr0=args.lr0, project=str(out_dir),
        name="train", exist_ok=True, plots=True, val=True, save=True,
        verbose=True,
    )
    LOGGER.info("Training arguments: %s", json.dumps(kwargs, indent=2,
                                                     default=str))
    started = time.time()
    model.train(**kwargs)
    elapsed = (time.time() - started) / 60
    LOGGER.info("Training finished in %.1f minutes.", elapsed)
    return elapsed


def copy_weights(train_dir: Path, weights_dir: Path) -> None:
    import shutil

    for name in ("best.pt", "last.pt"):
        src = train_dir / "weights" / name
        if src.exists():
            shutil.copy2(src, weights_dir / name)
