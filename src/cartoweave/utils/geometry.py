from __future__ import annotations
import math
import numpy as np
from typing import Sequence, Union
from .kernels import softabs, EPS_ABS, EPS_NORM

Array = np.ndarray

def project_point_to_segment(cx, cy, ax, ay, bx, by):
    vx, vy = (bx-ax), (by-ay)
    L2 = vx*vx + vy*vy
    if L2 <= 1e-18:
        return ax, ay, 0.0, 1.0, 0.0
    t = ((cx-ax)*vx + (cy-ay)*vy) / L2
    t = max(0.0, min(1.0, t))
    qx = ax + t*vx; qy = ay + t*vy
    L  = math.sqrt(L2) + EPS_NORM
    return qx, qy, t, vx/L, vy/L

def poly_signed_area(poly_xy: np.ndarray) -> float:
    a = 0.0
    n = poly_xy.shape[0]
    for i in range(n):
        x1,y1 = poly_xy[i]
        x2,y2 = poly_xy[(i+1)%n]
        a += x1*y2 - x2*y1
    return 0.5*a

def rect_half_extent_along_dir(w: float, h: float, nx: float, ny: float, eps_abs: float = EPS_ABS) -> float:
    return 0.5*w*softabs(nx, eps_abs) + 0.5*h*softabs(ny, eps_abs)


def polylines_to_segments(lines: Union[Sequence[Array], Array], eps: float = 1e-12) -> Array:
    """
    Accepts:
      - list[np.ndarray(Mi,2)]  (preferred)
      - np.ndarray(Nl,2,2)      (legacy segments)
      - np.ndarray(Nl,M,2)      (fixed-length polylines)
      - np.ndarray(M,2)         (single polyline)
    Returns a numeric (S,2,2) float array with zero-length segments dropped.
    """
    arr = np.asarray(lines, dtype=object)

    def _from_polyline(P: Array) -> Array:
        P = np.asarray(P, float).reshape(-1, 2)
        if P.shape[0] < 2:
            return np.zeros((0, 2, 2), float)
        AB = np.stack([P[:-1], P[1:]], axis=1)
        d = np.linalg.norm(AB[:, 1] - AB[:, 0], axis=1)
        return AB[d > eps]

    if arr.dtype == object:
        chunks = [_from_polyline(p) for p in lines]
        return np.concatenate(chunks, axis=0) if len(chunks) else np.zeros((0, 2, 2), float)

    if arr.ndim == 3:
        if arr.shape[-2:] == (2, 2):
            AB = np.asarray(arr, float)
            d = np.linalg.norm(AB[:, 1] - AB[:, 0], axis=1)
            return AB[d > eps]
        chunks = [_from_polyline(p) for p in np.asarray(lines)]
        return np.concatenate(chunks, axis=0) if len(chunks) else np.zeros((0, 2, 2), float)

    if arr.ndim == 2 and arr.shape[1] == 2:
        return _from_polyline(arr)

    return np.zeros((0, 2, 2), float)
