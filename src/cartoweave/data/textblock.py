from __future__ import annotations

import string
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "random_alpha_string",
    "random_text_lines",
    "load_font",
    "measure_text_block",
]

ALPHABET = string.ascii_letters


def random_alpha_string(rng: np.random.Generator, length: int) -> str:
    """Return a random alphabetic string of ``length`` characters."""
    chars = rng.choice(list(ALPHABET), size=int(length))
    return "".join(chars.tolist())


def random_text_lines(rng: np.random.Generator, len_range: Tuple[int, int]) -> list[str]:
    """Generate either one or three random strings within ``len_range``."""
    lo, hi = map(int, len_range)
    lengths = rng.integers(lo, hi + 1, size=3)
    strings = [random_alpha_string(rng, int(L)) for L in lengths]
    if rng.random() < 0.5:
        return [strings[0]]
    return strings


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back to PIL's default on failure."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # pragma: no cover - IOError/OSError variants
        return ImageFont.load_default()


def _measure_line_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
    else:  # pragma: no cover - older Pillow versions
        img = Image.new("L", (1, 1), 0)
        draw = ImageDraw.Draw(img)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def measure_text_block(
    lines: Iterable[str],
    font: ImageFont.FreeTypeFont,
    line_spacing_px: int,
    padding_x: int,
    padding_y: int,
) -> tuple[int, int]:
    """Measure width/height for ``lines`` using ``font`` and spacing/padding."""
    widths = []
    heights = []
    for line in lines:
        w, h = _measure_line_bbox(line, font)
        widths.append(w)
        heights.append(h)
    if not widths:
        max_w = 0
        total_h = 0
    else:
        max_w = max(widths)
        total_h = sum(heights)
    n = len(widths)
    total_h += max(0, n - 1) * int(line_spacing_px)
    W = max(1, int(max_w + 2 * padding_x))
    H = max(1, int(total_h + 2 * padding_y))
    return W, H
