from __future__ import annotations
import math
import numpy as np
from typing import Sequence, Union
from .kernels import (
    softabs,
    softplus,
    sigmoid,
    smoothmax,
    softclip,
    d_softclip,
    EPS_ABS,
    EPS_NORM,
)

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


def project_point_to_segment_diff(cx, cy, ax, ay, bx, by, beta: float = 8.0):
    """Differentiable point-to-segment projection returning Jacobian.

    The segment ``A(ax,ay)`` to ``B(bx,by)`` is parameterised by ``t`` in
    ``[0,1]`` using a soft clip so both the projected point and its derivative
    with respect to ``C(cx,cy)`` remain smooth near the end caps.

    Returns
    -------
    tuple
        ``(qx, qy, t, tx, ty, J)`` where ``(qx,qy)`` is the projection,
        ``t`` is the clipped parameter, ``(tx,ty)`` the unit tangent and
        ``J`` the ``2×2`` Jacobian ``∂q/∂C``.
    """

    vx, vy = (bx - ax), (by - ay)
    L2 = vx * vx + vy * vy
    if L2 <= 1e-18:
        return ax, ay, 0.0, 1.0, 0.0, np.zeros((2, 2), float)

    w_cx = cx - ax
    w_cy = cy - ay
    t_un = (w_cx * vx + w_cy * vy) / L2
    dt_un = np.array([vx / L2, vy / L2], float)
    t = softclip(t_un, 0.0, 1.0, beta=beta)
    dsoft = d_softclip(t_un, 0.0, 1.0, beta=beta)
    dt = dsoft * dt_un

    qx = ax + t * vx
    qy = ay + t * vy
    dq_dc = np.array([[vx * dt[0], vx * dt[1]], [vy * dt[0], vy * dt[1]]], float)

    L = math.sqrt(L2) + EPS_NORM
    tx, ty = vx / L, vy / L
    return qx, qy, t, tx, ty, dq_dc

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


def segment_rect_gate(
    A, B, C, wh,
    min_gap=0.0,
    alpha=8.0,       # normal softplus sharpness
    eta=6.0,         # tangent sigmoid scale
    cap_scale=1.0,   # u_cap = cap_scale*max(w/2, h/2)
    g_min_int=0.1,   # floor when projection falls within segment
    kappa=8.0,       # how strongly 'inside-segment' lifts the floor
    beta=8.0,        # smoothmax sharpness
    *,
    params: dict | None = None,
):
    """
    Continuous gate g∈(0,1] approximating the old boolean:
    - Normal gate from soft penetration margin p_n = softplus((r_n+min_gap) - |s|, alpha)
    - Tangent gate from sigmoid on |u| vs u_cap
    - Floor 'g_min_int' when the projection is interior, via a smooth interiorness weight.
    Returns: (p_n, g, extra_dict) where p_n≥0, g∈(0,1].
    """
    Ax, Ay = float(A[0]), float(A[1])
    Bx, By = float(B[0]), float(B[1])
    Cx, Cy = float(C[0]), float(C[1])
    w, h = float(wh[0]), float(wh[1])

    dx, dy = (Bx - Ax), (By - Ay)
    L2 = dx * dx + dy * dy
    L = math.sqrt(L2)

    params = params or {}
    L_floor = float(params.get("L_floor", 1e-3))
    if L < L_floor:
        extras = dict(
            L=L,
            s=0.0,
            u=0.0,
            r_n=0.0,
            u_cap=0.0,
            g_tan=0.0,
            pi_in=0.0,
            t=0.0,
            ux=0.0,
            uy=0.0,
            nx=0.0,
            ny=0.0,
        )
        return 0.0, 0.0, extras

    invL = 1.0 / max(L, L_floor)
    ux, uy = (dx * invL, dy * invL)
    nx, ny = (-uy, ux)

    qx, qy = (Cx - Ax), (Cy - Ay)
    s = qx*nx + qy*ny
    u = qx*ux + qy*uy

    abs_s = softabs(s, EPS_ABS)
    abs_u = softabs(u, EPS_ABS)

    r_n = rect_half_extent_along_dir(w, h, nx, ny)
    p_n = softplus((r_n + min_gap) - abs_s, alpha)

    u_cap = cap_scale * max(w * 0.5, h * 0.5)
    g_tan = sigmoid(-(abs_u - u_cap) / max(eta, 1e-9))

    t = (u * invL + 0.5)
    abs_2t1 = softabs(2.0 * t - 1.0, EPS_ABS)
    bell = 1.0 - abs_2t1
    pi_in = sigmoid(kappa * bell)

    g_floor = g_min_int * pi_in
    g = smoothmax(g_tan, g_floor, beta=beta)

    dpi_dt = pi_in * (1.0 - pi_in) * kappa * (-2.0 * (2.0 * t - 1.0) / max(abs_2t1, EPS_ABS))
    dpi_du = dpi_dt * invL
    deriv_clip = float(params.get("deriv_clip", 1.0e3))
    if dpi_du > deriv_clip:
        dpi_du = deriv_clip
    elif dpi_du < -deriv_clip:
        dpi_du = -deriv_clip

    extras = dict(
        L=L,
        s=s,
        u=u,
        r_n=r_n,
        u_cap=u_cap,
        g_tan=g_tan,
        pi_in=pi_in,
        t=t,
        ux=ux,
        uy=uy,
        nx=nx,
        ny=ny,
        dpi_du=dpi_du,
    )
    return p_n, g, extras


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
