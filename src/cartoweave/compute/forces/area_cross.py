# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import softplus, sigmoid, smoothmax, softabs, EPS_DIST, EPS_NORM, EPS_ABS
from cartoweave.utils.geometry import poly_signed_area, segment_rect_gate
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger

def _val(lab, key, default=None):
    """通用字段读取：兼容 dict 和 LabelState。
       - 支持 'kind' / 'mode' / 其它 meta 字段（mode 会从 meta 提升）
    """
    if isinstance(lab, dict):
        if key == "mode":
            return lab.get("mode") or (lab.get("meta") or {}).get("mode", default)
        return lab.get(key, default)
    if key == "mode":
        m = getattr(lab, "meta", None)
        return (m or {}).get("mode", default)
    return getattr(lab, key, default)


def _WH(lab):
    """统一尺寸读取：返回 np.array([w, h])。"""
    v = lab["WH"] if isinstance(lab, dict) else getattr(lab, "WH", None)
    return np.asarray(v, dtype=float)


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

"""Boolean AABB pre-gate replaced by continuous gate built from normal soft
penetration and tangent sigmoid with smooth floor; parameters (min_gap, alpha,
eta, cap_scale, g_min_int) unchanged."""


def _legacy_aabb_gate(ax, ay, bx, by, cx, cy, w, h, pad=0.0):
    x1, x2 = (ax, bx) if ax <= bx else (bx, ax)
    y1, y2 = (ay, by) if ay <= by else (by, ay)
    rx1, rx2 = cx - w * 0.5 - pad, cx + w * 0.5 + pad
    ry1, ry2 = cy - h * 0.5 - pad, cy + h * 0.5 + pad
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)


@register("area.cross")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("area.cross", cfg, 0.0)
    if phase != "pre_anchor" or w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.cross"}

    labels_all = scene.get("labels", [])
    areas = scene.get("areas", [])
    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [_val(lab, "mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]
    skip_circle = int(np.count_nonzero(~mask))

    k_cross = float(cfg.get("area.k.cross", 900.0))
    min_gap = float(cfg.get("area.cross.min_gap", 1.5))
    eta_tan = float(cfg.get("area.cross.eta", 2.0))
    alpha_sp = float(cfg.get("area.cross.alpha", 0.35))
    cap_scale = float(cfg.get("area.cross.tan_cap_scale", 1.0))
    use_lc = bool(cfg.get("area.cross.use_logcosh", True))
    p0_lc = float(cfg.get("area.cross.sat_p0", 2.0))
    g_min_int = float(cfg.get("area.cross.gate_min_interior", 0.6))
    eps_abs = float(cfg.get("eps.abs", EPS_ABS))
    kappa = float(cfg.get("area.cross.kappa", 8.0))
    beta_smax = float(cfg.get("area.cross.beta_smax", 8.0))

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]

    use_legacy_gate = bool(cfg.get("area.cross.use_legacy_gate", False) or os.getenv("AREA_CROSS_USE_LEGACY_GATE"))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = _anchor(lab)
        own_idx = int(a["index"]) if a and a["kind"] == "area" and a["index"] is not None else -1
        if w_i <= 0.0 and h_i <= 0.0:
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for ai, A in enumerate(areas):
            if ai == own_idx:
                continue
            poly = A.get("polygon", None)
            if poly is None:
                continue
            arr = np.asarray(poly, float).reshape(-1, 2)
            if arr.shape[0] < 3:
                continue
            ccw = (poly_signed_area(arr) > 0.0)
            fx_sum = fy_sum = 0.0
            best = 0.0
            n = arr.shape[0]
            for k in range(n):
                ax, ay = float(arr[k, 0]), float(arr[k, 1])
                bx, by = float(arr[(k + 1) % n, 0]), float(arr[(k + 1) % n, 1])
                if ccw:
                    A_seg, B_seg = (ax, ay), (bx, by)
                else:
                    A_seg, B_seg = (bx, by), (ax, ay)
                p_n, g_soft, ex = segment_rect_gate(
                    A=A_seg,
                    B=B_seg,
                    C=(cx, cy),
                    wh=(w_i, h_i),
                    min_gap=min_gap,
                    alpha=alpha_sp,
                    eta=eta_tan,
                    cap_scale=cap_scale,
                    g_min_int=g_min_int,
                    kappa=kappa,
                    beta=beta_smax,
                )
                s = ex["s"]
                u = ex["u"]
                L = ex["L"]
                nx_in, ny_in = ex["nx"], ex["ny"]
                tx, ty = ex["ux"], ex["uy"]
                g_tan = ex["g_tan"]
                pi_in = ex["pi_in"]
                t = ex["t"]
                g_floor = g_min_int * pi_in

                if use_legacy_gate:
                    hit = _legacy_aabb_gate(ax, ay, bx, by, cx, cy, w_i, h_i, pad=min_gap)
                    g = 1.0 if hit else g_soft
                else:
                    hit = False
                    g = g_soft

                abs_u = softabs(u, eps_abs)
                abs_s = softabs(s, eps_abs)

                x_gp = g * p_n
                if use_lc:
                    denom = max(p0_lc, eps)
                    t0 = x_gp / denom
                    E_k = k_cross * (
                        denom * (abs(t0) + math.log1p(math.exp(-2.0 * abs(t0))) - math.log(2.0))
                    )
                    dEdx = k_cross * math.tanh(t0)
                else:
                    E_k = 0.5 * k_cross * (x_gp * x_gp)
                    dEdx = k_cross * x_gp

                if use_legacy_gate and hit:
                    dgx = dgy = 0.0
                else:
                    coeff_u = u / max(abs_u, eps)
                    g_tan_prime = g_tan * (1.0 - g_tan) * (-1.0 / max(eta_tan, eps)) * coeff_u
                    abs_2t1 = softabs(2.0 * t - 1.0, eps_abs)
                    coeff_2t1 = (2.0 * t - 1.0) / max(abs_2t1, eps)
                    dpi_dt = pi_in * (1.0 - pi_in) * kappa * (-2.0 * coeff_2t1)
                    dpi_du = dpi_dt * (1.0 / max(L, eps))
                    g_floor_prime = g_min_int * dpi_du
                    sigma = sigmoid(beta_smax * (g_tan - g_floor))
                    dg_du = sigma * g_tan_prime + (1.0 - sigma) * g_floor_prime
                    dgx, dgy = dg_du * tx, dg_du * ty

                coeff_s = s / max(abs_s, eps)
                sig_az = sigmoid(alpha_sp * ((ex["r_n"] + min_gap) - abs_s))
                dpx = sig_az * (-coeff_s) * nx_in
                dpy = sig_az * (-coeff_s) * ny_in

                dx_dc_x = p_n * dgx + g * dpx
                dx_dc_y = p_n * dgy + g * dpy
                fx_k = -(dEdx * dx_dc_x)
                fy_k = -(dEdx * dx_dc_y)
                fx_sum += fx_k
                fy_sum += fy_k
                best = max(best, abs(dEdx))
                E += E_k
            F[i, 0] += fx_sum
            F[i, 1] += fy_sum
            S[i].append((int(ai), float(fx_sum), float(fy_sum), float(best)))

    logger.debug("term_area_cross: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"term": "area.cross", "area_cross": S}
