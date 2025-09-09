# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
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
    float_param,
    active_element_indices,
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


@register("pl.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "pl", "rect")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}
    pts_all = scene.get("points") or []
    if len(pts_all) == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}

    labels = read_labels_aligned(scene, P)
    active_pts = active_element_indices(labels, "point")
    pts = [pts_all[i] for i in sorted(active_pts) if 0 <= i < len(pts_all)]
    if not pts:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}
    pts = np.asarray(pts, float).reshape(-1, 2)
    N = int(P.shape[0])
    M = pts.shape[0]
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]
    WH = normalize_WH_from_labels(labels, N, "pl.rect")

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
        for j in range(M):
            x, y = float(pts[j, 0]), float(pts[j, 1])
            dx, dy = cx - x, cy - y
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * w_i
            sy = ady - 0.5 * h_i

            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5

            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_ddx = (spx / g) * sigmoid(beta_sep * sx) * (dx / (adx + eps))
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * (dy / (ady + eps))
                    fx = -dEdg * d_g_ddx
                    fy = -dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[i, 0] += fx
                F[i, 1] += fy

            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                fx_in = -k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                fy_in = -k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                F[i, 0] += fx_in
                F[i, 1] += fy_in

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "pl.rect", "pl": int(N * M)}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``pl.rect`` field caused by static points."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    pts = scene.get("points")
    if pts is None or len(pts) == 0:
        return np.zeros_like(xy, float)

    k_out = float_param(params, "k_out", 0.8)
    pts = np.asarray(pts, float).reshape(-1, 2)

    dx = xy[:, None, 0] - pts[None, :, 0]
    dy = xy[:, None, 1] - pts[None, :, 1]
    dist_sq = dx * dx + dy * dy + 1e-9
    dist = np.sqrt(dist_sq)
    mag = k_out / dist_sq
    fx = (mag * dx / dist).sum(axis=1)
    fy = (mag * dy / dist).sum(axis=1)
    F = np.stack([fx, fy], axis=1)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("pl.rect")(probe)
