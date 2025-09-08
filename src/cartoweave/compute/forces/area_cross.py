# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.kernels import softplus, sigmoid, softabs
from cartoweave.utils.geometry import poly_signed_area, segment_rect_gate
from cartoweave.utils.shape import as_nx2
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
    poly_as_array,
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
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "area", "cross")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.cross"}

    areas      = scene.get("areas", []) or []

    N = int(P.shape[0])
    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "area.cross")

    k_cross = float(900.0 if tc.get("k_cross") is None else tc.get("k_cross"))
    ker = kernel_params(
        tc,
        defaults={"model": "logcosh", "gate_gamma": 8.0, "exponent": 1.0, "soft_eps": 1e-6},
    )
    min_gap = float(1.5 if tc.get("min_gap") is None else tc.get("min_gap"))
    eta_tan = float(2.0 if tc.get("eta") is None else tc.get("eta"))
    alpha_sp = float(0.35 if tc.get("alpha") is None else tc.get("alpha"))
    cap_scale = float(1.0 if tc.get("tan_cap_scale") is None else tc.get("tan_cap_scale"))
    use_lc = bool(tc.get("use_logcosh") if tc.get("use_logcosh") is not None else True)
    p0_lc = float(2.0 if tc.get("sat_p0") is None else tc.get("sat_p0"))
    g_min_int = float(0.6 if tc.get("gate_min_interior") is None else tc.get("gate_min_interior"))
    eps_abs = epss["eps_abs"]
    kappa = float(8.0 if tc.get("kappa") is None else tc.get("kappa"))
    beta_smax = float(8.0 if tc.get("beta_smax") is None else tc.get("beta_smax"))

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]

    use_legacy_gate = bool(tc.get("use_legacy_gate", False) or os.getenv("AREA_CROSS_USE_LEGACY_GATE"))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = _anchor(lab)
        own_idx = int(a["index"]) if a and a["kind"] == "area" and a["index"] is not None else -1
        if w_i <= 0.0 and h_i <= 0.0:
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for ai, poly in enumerate(areas):
            if ai == own_idx:
                continue
            #poly = A.get("polygon", None)
            arr = poly_as_array(poly)
            if arr is None:
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

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "area.cross", "area_cross": S}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample the ``area.cross`` field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    areas = scene.get("areas", []) or []
    if not areas:
        return np.zeros_like(xy, float)

    k_cross = float_param(params, "k_cross", 900.0)
    min_gap = float_param(params, "min_gap", 1.5)
    eta_tan = float_param(params, "eta", 2.0)
    alpha_sp = float_param(params, "alpha", 0.35)
    cap_scale = float_param(params, "tan_cap_scale", 1.0)
    use_lc = bool(params.get("use_logcosh", True))
    p0_lc = float_param(params, "sat_p0", 2.0)
    g_min_int = float_param(params, "gate_min_interior", 0.6)
    kappa = float_param(params, "kappa", 8.0)
    beta_smax = float_param(params, "beta_smax", 8.0)
    eps_abs = float_param(params, "eps_abs", 1e-3)
    eps = float_param(params, "eps_numeric", 1e-12)

    F = np.zeros_like(xy, float)

    for poly in areas:
        arr = poly_as_array(poly)
        if arr is None:
            continue
        n = arr.shape[0]
        if n < 2:
            continue
        ccw = (poly_signed_area(arr) > 0.0)
        for i in range(xy.shape[0]):
            px, py = float(xy[i, 0]), float(xy[i, 1])
            fx_sum = fy_sum = 0.0
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
                    C=(px, py),
                    wh=(0.0, 0.0),
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
                g = g_soft
                abs_u = softabs(u, eps_abs)
                abs_s = softabs(s, eps_abs)
                x_gp = g * p_n
                if use_lc:
                    denom = max(p0_lc, eps)
                    t0 = x_gp / denom
                    dEdx = k_cross * math.tanh(t0)
                else:
                    dEdx = k_cross * x_gp
                coeff_u = u / max(abs_u, eps)
                g_tan_prime = g_tan * (1.0 - g_tan) * (-1.0 / max(eta_tan, eps)) * coeff_u
                abs_2t1 = softabs(2.0 * t - 1.0, eps_abs)
                coeff_2t1 = (2.0 * t - 1.0) / max(abs_2t1, eps)
                dpi_dt = pi_in * (1.0 - pi_in) * kappa * (-2.0 * coeff_2t1)
                dpi_du = dpi_dt * (1.0 / max(L, eps))
                g_floor_prime = g_min_int * dpi_du
                sigma = sigmoid(beta_smax * (g_tan - g_min_int * pi_in))
                dg_du = sigma * g_tan_prime + (1.0 - sigma) * g_floor_prime
                dgx, dgy = dg_du * tx, dg_du * ty
                coeff_s = s / max(abs_s, eps)
                sig_az = sigmoid(alpha_sp * ((ex["r_n"] + min_gap) - abs_s))
                dpx = sig_az * (-coeff_s) * nx_in
                dpy = sig_az * (-coeff_s) * ny_in
                dx_dc_x = p_n * dgx + g * dpx
                dx_dc_y = p_n * dgy + g * dpy
                fx_sum += -(dEdx * dx_dc_x)
                fy_sum += -(dEdx * dx_dc_y)
            F[i, 0] += fx_sum
            F[i, 1] += fy_sum

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("area.cross")(probe)
