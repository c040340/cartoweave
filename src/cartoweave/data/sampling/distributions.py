"""Probability distributions used in sampling."""
from __future__ import annotations

import numpy as np


def sample_split_normal_trunc(
    rng: np.random.Generator,
    mean: float,
    sigma_left: float,
    sigma_right: float,
    low: float,
    high: float,
    size: int | tuple = (),
) -> np.ndarray:
    """Sample from a split normal distribution with truncation.

    The distribution is composed of two half Gaussians joined at ``mean`` with
    standard deviations ``sigma_left`` and ``sigma_right``.  Samples outside the
    interval ``[low, high]`` are rejected.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    size = tuple(np.atleast_1d(size).astype(int))
    out = np.empty(size, dtype=float)
    flat = out.reshape(-1)
    p_left = sigma_left / (sigma_left + sigma_right)
    for i in range(flat.size):
        for _ in range(128):
            if rng.random() < p_left:
                x = mean - abs(rng.normal(0.0, sigma_left))
            else:
                x = mean + abs(rng.normal(0.0, sigma_right))
            if low <= x <= high:
                flat[i] = x
                break
        else:
            flat[i] = np.clip(mean, low, high)
    return out

