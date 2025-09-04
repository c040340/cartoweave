from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np

from ..utils.geometry import poly_signed_area
from ..utils.kernels import EPS_DIST, EPS_AREA


def poly_centroid(poly: np.ndarray) -> Tuple[float, float]:
    """Return area-weighted centroid of *poly*.

    Falls back to vertex mean when the signed area is nearly zero. Supports
    polygons in either winding order.
    """
    pts = np.asarray(poly, float)
    if pts.shape[0] == 0:
        return 0.0, 0.0
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = pts.shape[0]
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        A += cross
        Cx += (x1 + x2) * cross
        Cy += (y1 + y2) * cross
    A *= 0.5
    if abs(A) <= EPS_AREA:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return float(Cx), float(Cy)


def project_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    """Project point ``P`` onto segment ``AB``.

    Returns ``(qx, qy, t)`` where ``Q`` is the closest point on ``AB`` and
    ``t`` is the segment parameter in ``[0,1]``.  Degenerate segments collapse to
    ``(A, t=0)``.
    """
    vx = x2 - x1
    vy = y2 - y1
    L2 = vx * vx + vy * vy
    if L2 <= EPS_DIST * EPS_DIST:
        return float(x1), float(y1), 0.0
    t = ((px - x1) * vx + (py - y1) * vy) / L2
    t = max(0.0, min(1.0, t))
    qx = x1 + t * vx
    qy = y1 + t * vy
    return float(qx), float(qy), float(t)


def project_point_to_polyline(px: float, py: float, polyline: np.ndarray) -> Tuple[float, float, int, float]:
    """Project point ``P`` onto ``polyline`` selecting the globally closest
    segment.
    """
    pts = np.asarray(polyline, float)
    m = max(pts.shape[0] - 1, 1)
    best_d2 = float("inf")
    best = (float("nan"), float("nan"), 0, 0.0)
    for i in range(m):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % pts.shape[0]]
        qx, qy, t = project_point_to_segment(px, py, x1, y1, x2, y2)
        d2 = (qx - px) * (qx - px) + (qy - py) * (qy - py)
        if d2 < best_d2 - EPS_DIST or (abs(d2 - best_d2) <= EPS_DIST and i > best[2]):
            best_d2 = d2
            best = (qx, qy, i, t)
    return best


def area_anchor_from_centroid_nearest_edge(poly: np.ndarray) -> Dict[str, object]:
    """Anchor polygon by projecting its centroid to the nearest edge."""
    pts = np.asarray(poly, float)
    cx, cy = poly_centroid(pts)
    best_i = 0
    best_t = 0.0
    best_qx = float("nan")
    best_qy = float("nan")
    best_d2 = float("inf")
    n = pts.shape[0]
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        qx, qy, t = project_point_to_segment(cx, cy, x1, y1, x2, y2)
        d2 = (qx - cx) * (qx - cx) + (qy - cy) * (qy - cy)
        if d2 < best_d2 - EPS_DIST or (abs(d2 - best_d2) <= EPS_DIST and i < best_i):
            best_i = i
            best_t = t
            best_qx = qx
            best_qy = qy
            best_d2 = d2
    x1, y1 = pts[best_i]
    x2, y2 = pts[(best_i + 1) % n]
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L <= EPS_DIST:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = dx / L, dy / L
    area = poly_signed_area(pts)
    if area >= 0.0:
        nx, ny = -ty, tx
    else:
        nx, ny = ty, -tx
    d = math.sqrt(max(best_d2, 0.0))
    return {
        "qx": float(best_qx),
        "qy": float(best_qy),
        "seg_index": int(best_i),
        "t": float(best_t),
        "tangent": (tx, ty),
        "normal_in": (nx, ny),
        "dist": float(d),
    }
