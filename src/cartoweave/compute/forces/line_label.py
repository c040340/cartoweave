# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.geometry import (
    project_point_to_segment,
    rect_half_extent_along_dir,
    polylines_to_segments,
)
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    invdist_energy,
    invdist_force_mag,
)
from cartoweave.utils.shape import as_nx2
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
)


def _anchor(lab):
    """统一锚读取：返回 dict {'kind': ..., 'index': ..., 't': ...} 或 None。"""
    if isinstance(lab, dict):
        a = lab.get("anchor")
        if a is None:
            return None
        return {
            "kind": a.get("kind") if isinstance(a, dict) else None,
            "index": a.get("index") if isinstance(a, dict) else None,
            "t": a.get("t") if isinstance(a, dict) else None,
        }
    a = getattr(lab, "anchor", None)
    if a is None:
        return None
    return {
        "kind": getattr(a, "kind", None),
        "index": getattr(a, "index", None),
        "t": getattr(a, "t", None),
    }


@register("ln.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "ln", "rect")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}
    segs_raw = scene.get("lines")
    if segs_raw is None or len(segs_raw) == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}
    segs_arr = polylines_to_segments(segs_raw)
    segs = segs_arr.reshape((segs_arr.shape[0], 4))
    N = int(P.shape[0])
    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]
    WH = normalize_WH_from_labels(labels, N, "ln.rect")
    F = np.zeros_like(P)
    E = 0.0

    k_out = float(0.8 if tc.get("k_out") is None else tc.get("k_out"))
    k_in = float(0.8 if tc.get("k_in") is None else tc.get("k_in"))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 2.0, "soft_eps": 1e-6})
    pwr = ker["kernel_exponent"]
    eps_d = ker["kernel_soft_eps"]
    eps_a = epss["eps_abs"]
    beta = tc.get("beta") or {}
    beta_sep = float(6.0 if beta.get("sep") is None else beta.get("sep"))
    beta_in = float(6.0 if beta.get("in") is None else beta.get("in"))
    g_eps = float(1e-6 if tc.get("g_eps") is None else tc.get("g_eps"))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for s in segs:
            ax, ay, bx, by = map(float, s)
            qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
            nx, ny = -ty, tx
            dx, dy = (cx - qx), (cy - qy)
            r_n = rect_half_extent_along_dir(w_i, h_i, nx, ny, eps_abs=eps_a)
            s_n = softabs(nx * dx + ny * dy, eps_a) - r_n
            s_t = softabs(tx * dx + ty * dy, eps_a)
            spn = softplus(s_n, beta_sep)
            spt = softplus(s_t, beta_sep)
            g = (spn * spn + spt * spt + g_eps * g_eps) ** 0.5
            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_dC_n = (spn / g) * sigmoid(beta_sep * s_n)
                    d_g_dC_t = (spt / g) * sigmoid(beta_sep * s_t)
                    coeff_n = (nx * dx + ny * dy) / (softabs(nx * dx + ny * dy, eps_a) + eps)
                    coeff_t = (tx * dx + ty * dy) / (softabs(tx * dx + ty * dy, eps_a) + eps)
                    dC_x = d_g_dC_n * coeff_n * nx + d_g_dC_t * coeff_t * tx
                    dC_y = d_g_dC_n * coeff_n * ny + d_g_dC_t * coeff_t * ty
                    fx = -dEdg * dC_x
                    fy = -dEdg * dC_y
                else:
                    fx = fy = 0.0
                F[i, 0] += fx
                F[i, 1] += fy

            if k_in > 0.0:
                vin = softplus(-s_n, beta_in)
                E += 0.5 * k_in * (vin * vin)
                coeff_n = (nx * dx + ny * dy) / (softabs(nx * dx + ny * dy, eps_a) + eps)
                dE_dC_n = k_in * vin * (-sigmoid(-beta_in * s_n)) * coeff_n
                fx_in = -dE_dC_n * nx
                fy_in = -dE_dC_n * ny
                F[i, 0] += fx_in
                F[i, 1] += fy_in

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "ln.rect", "ln": int(segs.shape[0])}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``ln.rect`` field from line segments acting on probes."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    segs_raw = scene.get("lines") or []
    if len(segs_raw) == 0:
        return np.zeros_like(xy, float)

    segs_arr = polylines_to_segments(segs_raw)
    segs = segs_arr.reshape((segs_arr.shape[0], 4))
    k_out = float(0.8 if params.get("k_out") is None else params.get("k_out"))

    F = np.zeros_like(xy, float)
    for ax, ay, bx, by in segs:
        ax = float(ax)
        ay = float(ay)
        bx = float(bx)
        by = float(by)
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey) + 1e-9
        tx, ty = ex / L, ey / L
        vx = xy[:, 0] - ax
        vy = xy[:, 1] - ay
        t = np.clip(vx * tx + vy * ty, 0.0, L)
        qx = ax + t * tx
        qy = ay + t * ty
        dx = xy[:, 0] - qx
        dy = xy[:, 1] - qy
        dist_sq = dx * dx + dy * dy + 1e-9
        dist = np.sqrt(dist_sq)
        mag = k_out / dist_sq
        F[:, 0] += mag * dx / dist
        F[:, 1] += mag * dy / dist

    if not np.isfinite(F).all():
        raise ValueError("ln.rect probe produced non-finite values")
    return F


register_probe("ln.rect")(probe)
