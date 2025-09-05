"""Polygon utilities and generators."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..sampling.helpers import inset_rect, project_to_rect_inset, frame_metrics


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if abs(o1) < eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) < eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) < eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) < eps and _on_segment(c, d, b, eps):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segments_self_intersect(poly: np.ndarray, eps: float = 1e-9) -> bool:
    """Return ``True`` if any two segments of ``poly`` intersect."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must be of shape (N,2)")
    pts = np.asarray(poly, float)
    n = pts.shape[0]
    if n < 4:
        return False
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1], eps):
                return True
    return False


def polygon_self_intersect(poly: np.ndarray, eps: float = 1e-9) -> bool:
    """Robust self-intersection test for polygons.

    ``poly`` may be open or closed; the closing edge is always assumed during
    the intersection test.
    """
    pts = np.asarray(poly, float)
    if pts.shape[0] < 4:
        return False
    return segments_self_intersect(pts, eps=eps)


def generate_polygon_by_area(
    rng: np.random.Generator,
    frame_size: tuple[int, int],
    S_target: float,
    inset_margin: float,
    min_edge_spacing: float,
    n_vertices_range: tuple[int, int] = (6, 14),
    max_tries: int = 32,
    eps: float = 1e-9,
) -> np.ndarray:
    """Generate a polygon with area roughly ``S_target``.

    The algorithm samples polar coordinates around a centre inside the inset
    rectangle.  If generation fails ``max_tries`` times a simple rectangular
    fallback is returned.
    """
    rect = inset_rect(frame_size, inset_margin)
    cx = 0.5 * (rect[0] + rect[2])
    cy = 0.5 * (rect[1] + rect[3])

    for _ in range(max_tries):
        k = int(rng.integers(n_vertices_range[0], n_vertices_range[1] + 1))
        angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, k))
        R0 = math.sqrt(abs(S_target) / math.pi)
        radii = R0 * rng.uniform(0.7, 1.3, k)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]
        pts += np.array([cx, cy])
        pts = project_to_rect_inset(pts, rect)
        # spacing check
        dists = np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1)
        if np.any(dists < min_edge_spacing - eps):
            continue
        # non-adjacent spacing
        for i in range(k):
            for j in range(i + 2, k):
                if j == k - 1 and i == 0:
                    continue
                if np.linalg.norm(pts[i] - pts[j]) < min_edge_spacing - eps:
                    break
            else:
                continue
            break
        else:
            if not polygon_self_intersect(pts, eps=eps):
                return pts.astype(float)
    # fallback rectangle
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

