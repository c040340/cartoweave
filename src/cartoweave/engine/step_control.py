"""Simplified step governor utilities.

These are **not** full solver implementations but provide lightweight
helpers for unit tests."""
from __future__ import annotations

import numpy as np


def apply_step_governor(prev: np.ndarray, proposed: np.ndarray, max_step: float) -> np.ndarray:
    """Clamp step so that ``|ΔP|_inf`` <= ``max_step``.

    Parameters
    ----------
    prev, proposed : np.ndarray
        Previous and proposed positions of shape (N, 2).
    max_step : float
        Maximum allowed per-axis step in pixels.
    """
    delta = proposed - prev
    inf_norm = np.max(np.abs(delta))
    if inf_norm <= max_step:
        return proposed
    scale = max_step / inf_norm
    return prev + delta * scale

__all__ = ["apply_step_governor"]
