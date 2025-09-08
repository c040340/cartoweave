from __future__ import annotations

import string
from typing import Iterable, Tuple, Sequence
from pathlib import Path

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


def _candidate_font_paths(path: str) -> Sequence[Path]:
    p = Path(path)
    here = Path(__file__).resolve()
    roots = [
        here.parents[2],              # repo root: .../src/cartoweave/data -> repo/
        here.parents[1],              # .../src/cartoweave/
        here.parents[0],              # .../src/
    ]
    # 1) 原样（若已是绝对路径或相对当前工作目录）
    yield p
    # 2) 相对源码树的 assets/
    for r in roots:
        yield (r / "assets" / p.name)
        yield (r / p)  # 以配置相对 repo root 的写法

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    last_err = None
    for cand in _candidate_font_paths(path):
        try:
            if cand.exists():
                return ImageFont.truetype(str(cand), size)
        except Exception as e:
            last_err = e
    # 如果你希望测试环境里“宁可降级也不要炸”，保留兜底，但明确记录：
    try:
        return ImageFont.truetype(str(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")), size)
    except Exception:
        # 最后兜底到位图字体，但注意部分 API 不支持
        return ImageFont.load_default()


def _measure_line_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    # 1) 首选 FreeType 的 getbbox
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    # 2) 再试 textbbox
    try:
        img = Image.new("L", (1, 1), 0)
        draw = ImageDraw.Draw(img)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        # 3) 最后兜底到 textsize（老 API，位图也支持）
        w, h = draw.textsize(text, font=font)
        return w, h


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
