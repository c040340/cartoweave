"""Lightweight text utilities used by scene generation."""
from __future__ import annotations

from dataclasses import dataclass
import string
import numpy as np


NON_ASCII_POOL = "示例文本汉字集合"


def rand_letters(rng: np.random.Generator, n_ascii: int, n_non_ascii: int) -> str:
    """Return a random string composed of ASCII and non-ASCII letters."""
    chars = []
    if n_ascii > 0:
        chars.append(rng.choice(list(string.ascii_uppercase)))
        chars.extend(rng.choice(list(string.ascii_letters), size=max(0, n_ascii - 1)))
    if n_non_ascii > 0:
        chars.extend(rng.choice(list(NON_ASCII_POOL), size=n_non_ascii))
    rng.shuffle(chars)
    return "".join(chars)


def measure_text_width(s: str, ascii_px: float, non_ascii_px: float) -> float:
    """Very rough text width estimate based on character classes."""
    ascii_count = sum(1 for c in s if ord(c) < 128)
    non_ascii_count = len(s) - ascii_count
    return ascii_count * ascii_px + non_ascii_count * non_ascii_px


@dataclass(frozen=True)
class LabelWidthSpec:
    """Estimated label widths for single and detail modes."""
    single_px: float
    detail_px: float
    font: str = "default"


def label_specs_for_len(label_len_hint: int, ascii_px: float = 8.0, non_ascii_px: float = 16.0) -> LabelWidthSpec:
    """Return width estimates for a label of approximately ``label_len_hint`` chars."""
    base = max(1, int(label_len_hint))
    single = base * ascii_px
    detail = (base + max(1, int(0.8 * base))) * ascii_px
    return LabelWidthSpec(single_px=float(single), detail_px=float(detail))

