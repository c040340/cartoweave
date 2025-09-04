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


def polyline_uniform_arc_centroid(vertices: np.ndarray, step_len: float = 4.0) -> np.ndarray:
    """Return centroid of uniformly arc-length-sampled points along a polyline.

    Parameters
    ----------
    vertices:
        ``(N,2)`` array of polyline vertices.
    step_len:
        Approximate spacing between samples along the arc.

    Degenerate cases (empty, single point or zero-length polylines) return the
    mean of available vertices.  All divisions guard against ``EPS_DIST``.
    """
    pts = np.asarray(vertices, float)
    if pts.shape[0] == 0:
        return np.zeros(2, float)
    if pts.shape[0] == 1:
        return pts[0].astype(float)
    seg_vec = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    total_len = float(seg_len.sum())
    if total_len <= EPS_DIST:
        return pts.mean(axis=0)

    # Sample distances along the polyline
    n_samples = max(int(np.ceil(total_len / max(step_len, EPS_DIST))) + 1, 2)
    dists = np.linspace(0.0, total_len, n_samples)
    cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))
    seg_idx = np.searchsorted(cumlen, dists, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)

    seg_start = pts[seg_idx]
    seg_delta = seg_vec[seg_idx]
    seg_len_sel = seg_len[seg_idx]
    t = (dists - cumlen[seg_idx]) / np.maximum(seg_len_sel, EPS_DIST)
    samples = seg_start + seg_delta * t[:, None]
    return samples.mean(axis=0)


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


def project_point_to_polyline(C: np.ndarray, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Project point ``C`` to the nearest point ``Q`` on a polyline.

    Returns ``(Q, tau, j)`` where ``tau`` is the unit tangent of the chosen
    segment and ``j`` is the segment index.  Computation is fully vectorised and
    guards divisions with ``EPS_DIST``.
    """
    pts = np.asarray(vertices, float)
    if pts.shape[0] <= 1:
        q = pts[0].astype(float) if pts.shape[0] == 1 else np.zeros(2, float)
        return q, np.array([1.0, 0.0], float), 0

    seg_start = pts[:-1]
    seg_vec = pts[1:] - seg_start
    seg_len2 = (seg_vec[:, 0] ** 2 + seg_vec[:, 1] ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((C[0] - seg_start[:, 0]) * seg_vec[:, 0] + (C[1] - seg_start[:, 1]) * seg_vec[:, 1]) / np.maximum(seg_len2, EPS_DIST)
    t = np.clip(t, 0.0, 1.0)
    proj = seg_start + seg_vec * t[:, None]
    diff = proj - C
    d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
    j = int(np.argmin(d2))
    dmin = d2[j]
    ties = np.where(np.abs(d2 - dmin) <= EPS_DIST)[0]
    if ties.size > 1:
        j = int(ties[-1])
    q = proj[j]
    vec = seg_vec[j]
    L = np.sqrt(seg_len2[j])
    if L <= EPS_DIST:
        tau = np.array([1.0, 0.0], float)
    else:
        tau = vec / L
    return q, tau, j


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
