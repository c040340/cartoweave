"""Utility geometry helpers for sampling routines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def frame_metrics(frame_size: tuple[int, int]) -> tuple[float, float]:
    """Return diagonal length and canvas area for ``frame_size``.

    Parameters
    ----------
    frame_size:
        Tuple ``(width, height)`` in pixels.

    Returns
    -------
    D, A: float
        Diagonal length and area of the canvas.
    """
    w, h = frame_size
    D = float((w * w + h * h) ** 0.5)
    A = float(w) * float(h)
    return D, A


def inset_rect(frame_size: tuple[int, int], margin_px: float) -> tuple[float, float, float, float]:
    """Return the coordinates of a rectangle inset by ``margin_px`` on all sides."""
    w, h = frame_size
    margin = float(margin_px)
    x0, y0 = margin, margin
    x1, y1 = float(w) - margin, float(h) - margin
    if x1 <= x0 or y1 <= y0:
        raise ValueError("margin too large for frame size")
    return x0, y0, x1, y1


def project_to_rect_inset(p: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Project a 2D point ``p`` into the rectangle ``rect``.

    If ``p`` lies outside ``rect`` it is clamped to the nearest boundary.  The
    function returns a new array and does not modify ``p`` in-place.
    """
    if p.shape[-1] != 2:
        raise ValueError("point must be of shape (...,2)")
    x0, y0, x1, y1 = rect
    x = np.clip(p[..., 0], x0, x1)
    y = np.clip(p[..., 1], y0, y1)
    return np.stack([x, y], axis=-1)

