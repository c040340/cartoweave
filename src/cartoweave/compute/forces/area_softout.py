# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    softclip,
    d_softclip,
)
from cartoweave.utils.geometry import (
    project_point_to_segment_diff,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.numerics import softmin_weights_with_grad
from cartoweave.utils.shape import as_nx2
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
    poly_as_array,
    anchor_info,
    active_element_indices,
)


@register("area.softout")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "area", "softout")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]
    L = P.shape[0] if P is not None else 0
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.softout"}

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
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.softout"}
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    active_mask = np.zeros(N, dtype=bool)
    active_mask[active_ids_arr] = True
    mask = base_mask & active_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "area.softout")

    k_push = float(250.0 if tc.get("k_push") is None else tc.get("k_push"))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 2.0, "soft_eps": 1e-6})
    min_gap = float(0.0 if tc.get("min_gap") is None else tc.get("min_gap"))
    beta = float(0.7 if tc.get("beta") is None else tc.get("beta"))
    alpha_sp = float(0.35 if tc.get("alpha") is None else tc.get("alpha"))
    gamma_out = float(0.5 if tc.get("outside_weight") is None else tc.get("outside_weight"))
    lambda_out = float(0.10 if tc.get("in_decay") is None else tc.get("in_decay"))
    lambda_in = float(0.06 if tc.get("out_decay") is None else tc.get("out_decay"))
    eps_abs = epss["eps_abs"]

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = anchor_info(lab)
        own_idx = int(a["index"]) if a and a["kind"] == "area" and a["index"] is not None else -1
        if w_i <= 0.0 or h_i <= 0.0:
            continue
        hx, hy = 0.5 * w_i, 0.5 * h_i
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for orig_ai, poly in areas:
            if orig_ai == own_idx:
                continue
            arr = poly_as_array(poly)
            if arr is None:
                continue
            m_list: list[float] = []
            dm_list: list[np.ndarray] = []
            ccw = (poly_signed_area(arr) > 0.0)
            nE = arr.shape[0]
            for k in range(nE):
                ax, ay = float(arr[k, 0]), float(arr[k, 1])
                bx, by = float(arr[(k + 1) % nE, 0]), float(arr[(k + 1) % nE, 1])
                qx, qy, _, tx, ty, J = project_point_to_segment_diff(cx, cy, ax, ay, bx, by)
                nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)
                dx, dy = (cx - qx), (cy - qy)
                s_k = nx_in * dx + ny_in * dy
                r_n = hx * softabs(nx_in, eps_abs) + hy * softabs(ny_in, eps_abs)
                m_k = s_k - r_n - min_gap
                m_list.append(m_k)
                n_vec = np.array([nx_in, ny_in], float)
                dm_k = n_vec - J.T @ n_vec
                dm_list.append(dm_k)
            if not m_list:
                continue
            v = np.asarray(m_list, float)
            wts, g = softmin_weights_with_grad(v, beta)
            m_eff = float((wts * v).sum())
            dm_eff = np.zeros(2, float)
            for gk, gradk in zip(g, dm_list):
                dm_eff += gk * gradk
            if np.linalg.norm(dm_eff) <= eps:
                continue
            if m_eff >= 0.0:
                r_in = softplus(m_eff, alpha_sp)
                s_in = sigmoid(alpha_sp * m_eff)
                # Hard clip replaced with softclip (C¹), keeps legacy rails (~[-80,80])
                t_in = softclip(lambda_in * m_eff, -80.0, 80.0, beta=8.0)
                decay_in = math.exp(-t_in)
                dt_dm = d_softclip(lambda_in * m_eff, -80.0, 80.0, beta=8.0) * lambda_in
                fmag = k_push * (r_in * s_in - (r_in * r_in) * dt_dm) * (decay_in ** 2)
                E += 0.5 * k_push * (r_in * decay_in) * (r_in * decay_in)
                mag = r_in * decay_in
            else:
                t_out = softclip(lambda_out * m_eff, -80.0, 80.0, beta=8.0)
                decay_out = math.exp(t_out)
                dt_dm = d_softclip(lambda_out * m_eff, -80.0, 80.0, beta=8.0) * lambda_out
                fmag = k_push * (gamma_out ** 2) * dt_dm * (decay_out ** 2)
                E += 0.5 * k_push * (gamma_out * decay_out) * (gamma_out * decay_out)
                mag = gamma_out * decay_out
            force = -fmag * dm_eff
            F[i, 0] += force[0]
            F[i, 1] += force[1]
            S[i].append((int(orig_ai), float(force[0]), float(force[1]), float(mag)))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "area.softout", "area_softout": S}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample the ``area.softout`` field at arbitrary world coordinates.

    The implementation mirrors :func:`evaluate` but treats ``xy`` points as
    zero-size probes that interact with area polygons defined in ``scene``.
    """

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    areas = scene.get("areas", []) or []
    if not areas:
        return np.zeros_like(xy, float)

    k_push = float_param(params, "k_push", 250.0)
    min_gap = float_param(params, "min_gap", 0.0)
    beta = float_param(params, "beta", 0.7)
    alpha_sp = float_param(params, "alpha", 0.35)
    gamma_out = float_param(params, "outside_weight", 0.5)
    lambda_out = float_param(params, "in_decay", 0.10)
    lambda_in = float_param(params, "out_decay", 0.06)

    F = np.zeros_like(xy, float)
    M = xy.shape[0]

    for poly in areas:
        arr = poly_as_array(poly)
        if arr is None:
            continue
        if arr.shape[0] < 3:
            continue
        ccw = (poly_signed_area(arr) > 0.0)
        nE = arr.shape[0]
        for i in range(M):
            px, py = float(xy[i, 0]), float(xy[i, 1])
            m_list: list[float] = []
            dm_list: list[np.ndarray] = []
            for k in range(nE):
                ax, ay = float(arr[k, 0]), float(arr[k, 1])
                bx, by = float(arr[(k + 1) % nE, 0]), float(arr[(k + 1) % nE, 1])
                qx, qy, _, tx, ty, J = project_point_to_segment_diff(px, py, ax, ay, bx, by)
                nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)
                dx, dy = px - qx, py - qy
                s_k = nx_in * dx + ny_in * dy - min_gap
                m_list.append(s_k)
                n_vec = np.array([nx_in, ny_in], float)
                dm_list.append(n_vec - J.T @ n_vec)
            if not m_list:
                continue
            v = np.asarray(m_list, float)
            wts, g = softmin_weights_with_grad(v, beta)
            m_eff = float((wts * v).sum())
            dm_eff = np.zeros(2, float)
            for gk, gradk in zip(g, dm_list):
                dm_eff += gk * gradk
            if np.linalg.norm(dm_eff) <= 1e-9:
                continue
            if m_eff >= 0.0:
                r_in = softplus(m_eff, alpha_sp)
                s_in = sigmoid(alpha_sp * m_eff)
                t_in = softclip(lambda_in * m_eff, -80.0, 80.0, beta=8.0)
                decay_in = math.exp(-t_in)
                dt_dm = d_softclip(lambda_in * m_eff, -80.0, 80.0, beta=8.0) * lambda_in
                fmag = k_push * (r_in * s_in - (r_in * r_in) * dt_dm) * (decay_in ** 2)
            else:
                t_out = softclip(lambda_out * m_eff, -80.0, 80.0, beta=8.0)
                decay_out = math.exp(t_out)
                dt_dm = d_softclip(lambda_out * m_eff, -80.0, 80.0, beta=8.0) * lambda_out
                fmag = k_push * (gamma_out ** 2) * dt_dm * (decay_out ** 2)
            F[i] += -fmag * dm_eff

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("area.softout")(probe)
