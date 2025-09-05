"""Poisson-disc sampling utilities."""
from __future__ import annotations

import math
from typing import List

import numpy as np


def poisson_disc(
    rng: np.random.Generator,
    width: float,
    height: float,
    r_min: float,
    k: int = 30,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Generate 2D Poisson-disc samples inside a rectangle.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    width, height : float
        Size of the sampling rectangle.
    r_min : float
        Minimum distance between samples.
    k : int, optional
        Candidates per active point, by default ``30``.
    x0, y0 : float, optional
        Origin of the rectangle, by default ``(0,0)``.
    """
    cell = r_min / math.sqrt(2.0)
    gw, gh = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    grid = [[-1] * gw for _ in range(gh)]
    samples: List[tuple[float, float]] = []
    active: List[int] = []

    def grid_ij(px: tuple[float, float]) -> tuple[int, int]:
        return int((px[1] - y0) // cell), int((px[0] - x0) // cell)

    def ok(px: tuple[float, float]) -> bool:
        gy, gx = grid_ij(px)
        for yy in range(max(gy - 2, 0), min(gy + 3, gh)):
            for xx in range(max(gx - 2, 0), min(gx + 3, gw)):
                si = grid[yy][xx]
                if si == -1:
                    continue
                qx, qy = samples[si]
                if (px[0] - qx) ** 2 + (px[1] - qy) ** 2 < r_min * r_min:
                    return False
        return True

    p0 = (rng.uniform(x0, x0 + width), rng.uniform(y0, y0 + height))
    samples.append(p0)
    active.append(0)
    gy, gx = grid_ij(p0)
    grid[gy][gx] = 0

    while active:
        a = rng.choice(active)
        base = samples[a]
        found = False
        for _ in range(k):
            th = rng.uniform(0.0, 2.0 * math.pi)
            rr = rng.uniform(r_min, 2.0 * r_min)
            cand = (
                base[0] + rr * math.cos(th),
                base[1] + rr * math.sin(th),
            )
            if x0 <= cand[0] < x0 + width and y0 <= cand[1] < y0 + height and ok(cand):
                samples.append(cand)
                active.append(len(samples) - 1)
                gy, gx = grid_ij(cand)
                grid[gy][gx] = len(samples) - 1
                found = True
                break
        if not found:
            active.remove(a)
    return np.asarray(samples, dtype=float)

