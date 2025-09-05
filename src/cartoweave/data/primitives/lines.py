"""Polyline generation utilities."""
from __future__ import annotations

import math
import numpy as np

from ..sampling.helpers import inset_rect, project_to_rect_inset

ANGLE_SIGMA = 0.35  # radians


def generate_polyline_by_length(
    rng: np.random.Generator,
    frame_size: tuple[int, int],
    L_target: float,
    min_vertex_spacing: float,
    inset_margin: float,
    segment_len_scale: float,
) -> np.ndarray:
    """Generate a random polyline inside the frame.

    The polyline is grown as a random walk with smooth angle changes until its
    total length reaches ``L_target``.  Vertices are kept inside an inset
    rectangle and spaced at least ``min_vertex_spacing`` apart.
    """
    rect = inset_rect(frame_size, inset_margin)
    x0, y0, x1, y1 = rect
    p = np.array([
        rng.uniform(x0, x1),
        rng.uniform(y0, y1),
    ])
    pts = [p]
    ang = rng.uniform(-math.pi, math.pi)
    total = 0.0
    while total < L_target:
        ang += rng.normal(0.0, ANGLE_SIGMA)
        step = max(min_vertex_spacing, rng.normal(segment_len_scale, 0.3 * segment_len_scale))
        q = p + step * np.array([math.cos(ang), math.sin(ang)])
        q = project_to_rect_inset(q, rect)
        if np.linalg.norm(q - p) < min_vertex_spacing:
            ang += math.pi / 2.0
            q = project_to_rect_inset(p + step * np.array([math.cos(ang), math.sin(ang)]), rect)
        total += float(np.linalg.norm(q - p))
        pts.append(q)
        p = q
    return np.asarray(pts, dtype=float)

