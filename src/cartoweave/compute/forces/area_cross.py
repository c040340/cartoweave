# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import logging
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.kernels import softplus, sigmoid, softabs
from cartoweave.utils.geometry import poly_signed_area, segment_rect_gate
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger
from ._common import (
    read_labels_aligned,
    get_mode,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
    poly_as_array,
    anchor_info,
    active_element_indices,
)

"""Rect–polygon crossing penalty with a continuous gate."""


def _aabb_gate(ax, ay, bx, by, cx, cy, w, h, pad=0.0) -> bool:
    """Cheap AABB rejection between segment AB and rect centred at C."""
    seg_min_x = min(ax, bx) - pad
    seg_max_x = max(ax, bx) + pad
    seg_min_y = min(ay, by) - pad
    seg_max_y = max(ay, by) + pad
    hw = w * 0.5 + pad
    hh = h * 0.5 + pad
    rect_min_x = cx - hw
    rect_max_x = cx + hw
    rect_min_y = cy - hh
    rect_max_y = cy + hh
    if seg_max_x < rect_min_x or seg_min_x > rect_max_x:
        return False
    if seg_max_y < rect_min_y or seg_min_y > rect_max_y:
        return False
    return True


@register("area.cross")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "area", "cross")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.cross"}

    labels = read_labels_aligned(scene, P)
    areas_all = scene.get("areas", []) or []
    N = int(P.shape[0])

    active_ids = scene.get("_active_ids_solver")
    if active_ids is None:
        active_ids_arr = np.arange(len(labels))
    else:
        active_ids_arr = np.asarray(active_ids, int)
    labels_active = [labels[i] for i in active_ids_arr if 0 <= i < len(labels)]
    active_area_indices = sorted(active_element_indices(labels_active, "area"))
    areas = [(idx, areas_all[idx]) for idx in active_area_indices if 0 <= idx < len(areas_all)]
    if not areas:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.cross"}

    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    active_mask = np.zeros(N, dtype=bool)
    active_mask[active_ids_arr] = True
    mask = base_mask & active_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "area.cross")

    k_cross = float(400.0 if tc.get("k_cross") is None else tc.get("k_cross"))
    ker = kernel_params(
        tc,
        defaults={"model": "logcosh", "gate_gamma": 8.0, "exponent": 1.0, "soft_eps": 1e-6},
    )
    min_gap = float(0.5 if tc.get("min_gap") is None else tc.get("min_gap"))
    eta_tan = float(
        (tc.get("tan_gate") or {}).get("eta", tc.get("eta", 4.0))
    )
    alpha_sp = float(0.7 if tc.get("alpha") is None else tc.get("alpha"))
    cap_scale = float(1.0 if tc.get("tan_cap_scale") is None else tc.get("tan_cap_scale"))
    use_lc = bool(tc.get("use_logcosh") if tc.get("use_logcosh") is not None else True)
    p0_lc = float(2.0 if tc.get("sat_p0") is None else tc.get("sat_p0"))
    g_min_int = float(0.0 if tc.get("gate_min_interior") is None else tc.get("gate_min_interior"))
    eps_abs = epss["eps_abs"]
    kappa = float(8.0 if tc.get("kappa") is None else tc.get("kappa"))
    beta_smax = float(4.0 if tc.get("beta_smax") is None else tc.get("beta_smax"))
    L_floor = float(1.0e-3 if tc.get("L_floor") is None else tc.get("L_floor"))
    deriv_clip = float(1.0e3 if tc.get("deriv_clip") is None else tc.get("deriv_clip"))
    _ = bool(tc.get("use_legacy_gate") if tc.get("use_legacy_gate") is not None else False)

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]
    min_edge_len = float("inf")
    skipped_short_edges = 0
    aabb_rejected_edges = 0
    max_force_norm = 0.0
    L_floor_sq = L_floor * L_floor

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = anchor_info(lab)
        own_idx = int(a["index"]) if a and a["kind"] == "area" and a["index"] is not None else -1
        if w_i <= 0.0 or h_i <= 0.0:
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for orig_ai, poly in areas:
            if orig_ai == own_idx:
                continue
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
                dx_seg = bx - ax
                dy_seg = by - ay
                L2 = dx_seg * dx_seg + dy_seg * dy_seg
                L = math.sqrt(L2)
                min_edge_len = min(min_edge_len, L)
                if L2 < L_floor_sq:
                    skipped_short_edges += 1
                    continue
                if not _aabb_gate(ax, ay, bx, by, cx, cy, w_i, h_i, pad=min_gap):
                    aabb_rejected_edges += 1
                    continue
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
                    params={"L_floor": L_floor, "deriv_clip": deriv_clip},
                )
                s = ex["s"]
                u = ex["u"]
                nx_in, ny_in = ex["nx"], ex["ny"]
                tx, ty = ex["ux"], ex["uy"]
                g_tan = ex["g_tan"]
                pi_in = ex["pi_in"]
                t = ex["t"]
                dpi_du = ex.get("dpi_du", 0.0)
                g_floor = g_min_int * pi_in
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

                coeff_u = u / max(abs_u, eps)
                g_tan_prime = g_tan * (1.0 - g_tan) * (-1.0 / max(eta_tan, eps)) * coeff_u
                g_floor_prime = g_min_int * dpi_du
                sigma = sigmoid(beta_smax * (g_tan - g_floor))
                dg_du = sigma * g_tan_prime + (1.0 - sigma) * g_floor_prime
                CLIP_FINAL = 1.0e4
                if dg_du > CLIP_FINAL:
                    dg_du = CLIP_FINAL
                elif dg_du < -CLIP_FINAL:
                    dg_du = -CLIP_FINAL
                dgx, dgy = dg_du * tx, dg_du * ty

                coeff_s = s / max(abs_s, eps)
                sig_az = sigmoid(alpha_sp * ((ex["r_n"] + min_gap) - abs_s))
                dpx = sig_az * (-coeff_s) * nx_in
                dpy = sig_az * (-coeff_s) * ny_in

                dx_dc_x = p_n * dgx + g * dpx
                dx_dc_y = p_n * dgy + g * dpy
                if dx_dc_x > CLIP_FINAL:
                    dx_dc_x = CLIP_FINAL
                elif dx_dc_x < -CLIP_FINAL:
                    dx_dc_x = -CLIP_FINAL
                if dx_dc_y > CLIP_FINAL:
                    dx_dc_y = CLIP_FINAL
                elif dx_dc_y < -CLIP_FINAL:
                    dx_dc_y = -CLIP_FINAL
                fx_k = -(dEdx * dx_dc_x)
                fy_k = -(dEdx * dx_dc_y)
                fx_sum += fx_k
                fy_sum += fy_k
                best = max(best, abs(dEdx))
                E += E_k
            F[i, 0] += fx_sum
            F[i, 1] += fy_sum
            S[i].append((int(orig_ai), float(fx_sum), float(fy_sum), float(best)))

    F = ensure_vec2(F, N)
    if F.size:
        max_force_norm = float(np.max(np.linalg.norm(F, axis=1)))
    min_edge = 0.0 if min_edge_len == float("inf") else float(min_edge_len)
    stats = {
        "min_edge_len": min_edge,
        "skipped_short_edges": int(skipped_short_edges),
        "aabb_rejected_edges": int(aabb_rejected_edges),
        "max_force_norm": float(max_force_norm),
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[area.cross] min_edge_len=%.4g, skipped_short=%d, aabb_reject=%d, maxF=%.4g",
            stats["min_edge_len"],
            stats["skipped_short_edges"],
            stats["aabb_rejected_edges"],
            stats["max_force_norm"],
        )
    return float(E), F, {"term": "area.cross", "area_cross": S, "stats": stats}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample the ``area.cross`` field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    areas = scene.get("areas", []) or []
    if not areas:
        return np.zeros_like(xy, float)

    k_cross = float_param(params, "k_cross", 400.0)
    min_gap = float_param(params, "min_gap", 0.5)
    eta_tan = float_param(params, "eta", 4.0)
    alpha_sp = float_param(params, "alpha", 0.7)
    cap_scale = float_param(params, "tan_cap_scale", 1.0)
    use_lc = bool(params.get("use_logcosh", True))
    p0_lc = float_param(params, "sat_p0", 2.0)
    g_min_int = float_param(params, "gate_min_interior", 0.0)
    kappa = float_param(params, "kappa", 8.0)
    beta_smax = float_param(params, "beta_smax", 4.0)
    L_floor = float_param(params, "L_floor", 1.0e-3)
    deriv_clip = float_param(params, "deriv_clip", 1.0e3)
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
                dx_seg = bx - ax
                dy_seg = by - ay
                if dx_seg * dx_seg + dy_seg * dy_seg < L_floor * L_floor:
                    continue
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
                    params={"L_floor": L_floor, "deriv_clip": deriv_clip},
                )
                s = ex["s"]
                u = ex["u"]
                nx_in, ny_in = ex["nx"], ex["ny"]
                tx, ty = ex["ux"], ex["uy"]
                g_tan = ex["g_tan"]
                pi_in = ex["pi_in"]
                t = ex["t"]
                dpi_du = ex.get("dpi_du", 0.0)
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
