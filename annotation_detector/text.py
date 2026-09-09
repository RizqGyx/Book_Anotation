"""Turning a plain-text book into page-sized chunks."""

from __future__ import annotations

import random
import re

from .constants import UNICODE_FALLBACKS

CHUNK_MAX_WORDS_V2 = 220
CHUNK_MAX_WORDS_V3 = 330


def normalize_text(text: str) -> str:
    """Map non-Latin-1 punctuation to ASCII so base-14 fonts can render it."""
    for src, dst in UNICODE_FALLBACKS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "ignore").decode("latin-1")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, good enough for book text."""
    text = re.sub(r"\s+", " ", normalize_text(text)).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_sentences(sentences: list[str], max_words: int) -> list[str]:
    """Group sentences into page-sized chunks."""
    chunks, current, count = [], [], 0
    for sentence in sentences:
        n_words = len(sentence.split())
        if current and count + n_words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_cycler(chunks: list[str], rng: random.Random):
    """Yield chunks in shuffled order, reshuffling once exhausted.

    Better than random.choice: the whole book is used evenly before anything
    repeats, which maximises text variety between samples.
    """
    order = list(range(len(chunks)))
    rng.shuffle(order)
    i = 0
    while True:
        if i >= len(order):
            rng.shuffle(order)
            i = 0
        index = order[i]
        i += 1
        yield index, chunks[index]


def weighted_choice(rng: random.Random, options, weights):
    return rng.choices(list(options), weights=list(weights), k=1)[0]


def fit_text_to_page(text_chunk: str, font_size: float,
                     base_words: int = CHUNK_MAX_WORDS_V3) -> str:
    """Trim text so a page fills up at any font size.

    Words that fit scale with the inverse square of font size, since both line
    height and glyph width grow with it.
    """
    budget = int(base_words * (9.0 / font_size) ** 1.9)
    words = text_chunk.split()
    return " ".join(words[:budget]) if len(words) > budget else text_chunk
