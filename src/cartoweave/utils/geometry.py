from __future__ import annotations
import math
import numpy as np
from .kernels import softabs, EPS_ABS, EPS_NORM

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
