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
)


@register("ll.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    N = int(P.shape[0])
    tc = term_cfg(cfg, "ll", "rect")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.rect"}

    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]

    # When the disk kernel is disabled we want all labels to participate in the
    # rectangle force irrespective of their declared ``ll_kernel``.  If both
    # kernels are enabled, respect the per‑label selection so that the two
    # implementations can coexist.
    disk_cfg = term_cfg(cfg, "ll", "disk")
    disk_enabled = bool(disk_cfg.get("enable", False))
    if disk_enabled:
        llk = [get_ll_kernel(l) for l in labels]
        rect_mask = np.array([(k or "") == "rect" for k in llk], dtype=bool)
    else:
        rect_mask = np.ones(N, dtype=bool)

    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask & rect_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "ll.rect")

    # ===== 你的原始主循环逻辑从这里继续（不变） =====
    F = np.zeros_like(P)
    E = 0.0

    k_out = float(0.3 if tc.get("k_out") is None else tc.get("k_out"))
    k_in = float(0.3 if tc.get("k_in") is None else tc.get("k_in"))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 2.0, "soft_eps": 1e-6})
    pwr = ker["kernel_exponent"]
    eps_d = ker["kernel_soft_eps"]
    eps_n = float(1e-9 if (tc.get("eps", {}).get("norm") is None) else tc.get("eps", {}).get("norm"))
    eps_a = epss["eps_abs"]

    beta = tc.get("beta") or {}
    beta_sep = float(6.0 if beta.get("sep") is None else beta.get("sep"))
    beta_in = float(6.0 if beta.get("in") is None else beta.get("in"))
    g_eps = float(1e-6 if tc.get("g_eps") is None else tc.get("g_eps"))

    for ai in range(len(idxs)):
        a = idxs[ai]
        xa, ya = float(P[a, 0]), float(P[a, 1])
        wa, ha = float(WH[a, 0]), float(WH[a, 1])
        for bi in range(ai + 1, len(idxs)):
            b  = idxs[bi]
            xb, yb = float(P[b, 0]), float(P[b, 1])
            wb, hb = float(WH[b, 0]), float(WH[b, 1])

            dx, dy = xa - xb, ya - yb
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * (wa + wb)
            sy = ady - 0.5 * (ha + hb)

            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g   = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5

            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    dspx_dsx = sigmoid(beta_sep * sx)
                    d_adx_ddx = dx / (adx + eps)
                    d_ady_ddy = dy / (ady + eps)
                    d_g_ddx = (spx / g) * dspx_dsx * d_adx_ddx
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * d_ady_ddy
                    fx = -dEdg * d_g_ddx
                    fy = -dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[a, 0] += fx; F[a, 1] += fy
                F[b, 0] -= fx; F[b, 1] -= fy

            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                c1 = k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                c2 = k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                fx_in = -c1; fy_in = -c2
                F[a, 0] += fx_in; F[a, 1] += fy_in
                F[b, 0] -= fx_in; F[b, 1] -= fy_in

    M = len(idxs)
    F = ensure_vec2(F, N)
    return float(E), F, {"term": "ll.rect", "pairs": int(M * (M - 1) // 2)}


def _pairwise_force_rect(src_xy: np.ndarray, src_wh: np.ndarray, xy: np.ndarray, params: dict) -> np.ndarray:
    """Force from a rectangular source at ``src_xy`` acting on probe points ``xy``."""

    k_out = float_param(params, "k_out", 0.3)
    dx = xy[:, 0] - src_xy[0]
    dy = xy[:, 1] - src_xy[1]
    dist_sq = dx * dx + dy * dy + 1e-6
    dist = np.sqrt(dist_sq)
    mag = k_out / dist_sq
    fx = -mag * dx / dist
    fy = -mag * dy / dist
    return np.stack([fx, fy], axis=1)


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample label-label rectangle field at ``xy`` points."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    labels_xy = np.asarray(scene.get("labels_xy", []), float)
    WH = (
        np.asarray(scene.get("WH"), float)
        if scene.get("WH") is not None
        else np.zeros((labels_xy.shape[0], 2), float)
    )
    if labels_xy.size == 0:
        return np.zeros_like(xy, float)

    F = np.zeros_like(xy, float)
    for p, wh in zip(labels_xy, WH):
        if not (np.isfinite(p).all() and np.isfinite(wh).all()):
            continue
        F += _pairwise_force_rect(np.asarray(p, float), np.asarray(wh, float), xy, params)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("ll.rect")(probe)

