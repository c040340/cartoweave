# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.geometry import (
    project_point_to_segment,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.numerics import (
    sigmoid_np,
    d_sigmoid_np,
    softabs_np,
    softmin_weights_np,
)
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
    poly_as_array,
    anchor_info,
)


@register("area.embed")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.embed"}
    tc = term_cfg(cfg, "area", "embed")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    eps = epss["eps_numeric"]

    labels = read_labels_aligned(scene, P)
    areas = scene.get("areas", [])
    N = int(P.shape[0])
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    active_ids = scene.get("_active_ids_solver")
    if active_ids is None:
        mask = base_mask
    else:
        active_mask = np.zeros(N, dtype=bool)
        active_mask[np.asarray(active_ids, int)] = True
        mask = base_mask & active_mask
    idxs = np.nonzero(mask)[0]
    WH = normalize_WH_from_labels(labels, N, "area.embed")

    k_embed = float(200.0 if tc.get("k_embed") is None else tc.get("k_embed"))
    k_tan = float(30.0 if tc.get("k_tan") is None else tc.get("k_tan"))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 2.0, "soft_eps": 1e-6})
    ratio_in = float(0.60 if tc.get("ratio_in") is None else tc.get("ratio_in"))
    beta_edge = float(6.0 if tc.get("beta_edge") is None else tc.get("beta_edge"))
    tan_gate = tc.get("tan_gate") or {}
    gate_eta = float(2.0 if tan_gate.get("eta") is None else tan_gate.get("eta"))
    gate_slack = float(1.0 if tan_gate.get("slack") is None else tan_gate.get("slack"))
    eps_abs = epss["eps_abs"]
    eps_div = eps

    F = np.zeros_like(P, float)
    E_total = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = anchor_info(lab)
        if not a or a["kind"] != "area":
            continue
        ai = int(a["index"]) if a["index"] is not None else -1
        if ai < 0 or ai >= len(areas):
            continue
        poly = areas[ai]
        arr = poly_as_array(poly)
        if arr is None:
            continue
        nE = arr.shape[0]
        if w_i <= 0.0 and h_i <= 0.0:
            continue
        hx, hy = 0.5 * w_i, 0.5 * h_i
        cx, cy = float(P[i, 0]), float(P[i, 1])
        ccw = (poly_signed_area(arr) > 0.0)

        s_list = np.zeros(nE)
        u_list = np.zeros(nE)
        rn_list = np.zeros(nE)
        n_list = np.zeros((nE, 2))
        t_list = np.zeros((nE, 2))
        q_list = np.zeros((nE, 2))
        tparam = np.zeros(nE)

        for k in range(nE):
            ax, ay = float(arr[k, 0]), float(arr[k, 1])
            bx, by = float(arr[(k + 1) % nE, 0]), float(arr[(k + 1) % nE, 1])
            qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
            nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)
            dx, dy = (cx - qx), (cy - qy)
            s_k = nx_in * dx + ny_in * dy
            u_k = tx * dx + ty * dy
            r_n_k = rect_half_extent_along_dir(w_i, h_i, nx_in, ny_in, eps_abs)
            s_list[k] = s_k
            u_list[k] = u_k
            rn_list[k] = r_n_k
            n_list[k] = (nx_in, ny_in)
            t_list[k] = (tx, ty)
            q_list[k] = (qx, qy)
            tparam[k] = t

        sabs = softabs_np(s_list, eps_abs)
        wgt = softmin_weights_np(sabs, beta_edge)
        J = (-beta_edge) * (np.diag(wgt) - np.outer(wgt, wgt))
        dv_ds = s_list / np.maximum(sabs, eps_div)
        s_star = (2.0 * ratio_in - 1.0) * rn_list
        ds = s_list - s_star
        E_perp = 0.5 * k_embed * (ds * ds)
        inv_eta = 1.0 / max(gate_eta, eps_div)
        x = ((rn_list + gate_slack) - sabs) * inv_eta
        g = sigmoid_np(x)
        if not (np.isfinite(g).all() and np.isfinite(wgt).all()):
            raise FloatingPointError("area_embed: non-finite gate/weights")
        E_tan = 0.5 * k_tan * g * (u_list * u_list)
        E_k = E_perp + E_tan
        E = float((wgt * E_k).sum())
        E_total += E

        dEperp_dC = (k_embed * ds)[:, None] * n_list
        gprime = d_sigmoid_np(x) * (-inv_eta) * dv_ds
        dEtan_dC = (
            (0.5 * k_tan * (u_list * u_list) * gprime)[:, None] * n_list
            + (k_tan * g * u_list)[:, None] * t_list
        )
        dEk_dC = dEperp_dC + dEtan_dC
        dv_dC = (dv_ds[:, None]) * n_list
        dw_dC = J @ dv_dC
        grad = (wgt[:, None] * dEk_dC).sum(axis=0) + (E_k[:, None] * dw_dC).sum(axis=0)
        F[i, 0] += -grad[0]
        F[i, 1] += -grad[1]
        k_min = int(np.argmin(sabs))
        S[i].append(
            (
                int(ai),
                float(-grad[0]),
                float(-grad[1]),
                float(E),
                k_min,
                float(s_list[k_min]),
                float(u_list[k_min]),
                float(wgt[k_min]),
            )
        )

    F = ensure_vec2(F, N)
    return float(E_total), F, {"term": "area.embed", "area_embed": S}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample the ``area.embed`` force field.

    This mirrors the analytic structure of :func:`evaluate` but omits terms
    involving label size, treating ``xy`` as zero-size probes.
    """

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    areas = scene.get("areas", []) or []
    if not areas:
        return np.zeros_like(xy, float)

    k_embed = float_param(params, "k_embed", 200.0)
    k_tan = float_param(params, "k_tan", 30.0)
    beta_edge = float_param(params, "beta_edge", 6.0)
    tan_gate = params.get("tan_gate") or {}
    gate_eta = float_param(tan_gate, "eta", 2.0)
    gate_slack = float_param(tan_gate, "slack", 1.0)
    eps_abs = float_param(params, "eps_abs", 1e-3)

    F = np.zeros_like(xy, float)
    M = xy.shape[0]

    for poly in areas:
        arr = poly_as_array(poly)
        if arr is None:
            continue
        nE = arr.shape[0]
        ccw = (poly_signed_area(arr) > 0.0)
        for i in range(M):
            px, py = float(xy[i, 0]), float(xy[i, 1])
            s_list: list[float] = []
            u_list: list[float] = []
            n_list: list[tuple[float, float]] = []
            t_list: list[tuple[float, float]] = []
            for k in range(nE):
                ax, ay = float(arr[k, 0]), float(arr[k, 1])
                bx, by = float(arr[(k + 1) % nE, 0]), float(arr[(k + 1) % nE, 1])
                qx, qy, _, tx, ty = project_point_to_segment(px, py, ax, ay, bx, by)
                nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)
                dx, dy = px - qx, py - qy
                s_k = nx_in * dx + ny_in * dy
                u_k = tx * dx + ty * dy
                s_list.append(s_k)
                u_list.append(u_k)
                n_list.append((nx_in, ny_in))
                t_list.append((tx, ty))
            if not s_list:
                continue
            s_arr = np.asarray(s_list, float)
            u_arr = np.asarray(u_list, float)
            n_arr = np.asarray(n_list, float)
            t_arr = np.asarray(t_list, float)
            sabs = softabs_np(s_arr, eps_abs)
            wgt = softmin_weights_np(sabs, beta_edge)
            ds = s_arr  # zero-size probe -> r_n = 0
            g = sigmoid_np(((gate_slack) - sabs) / max(gate_eta, 1e-9))
            grad = (k_embed * ds)[:, None] * n_arr + (k_tan * g * u_arr)[:, None] * t_arr
            F[i] += -(wgt[:, None] * grad).sum(axis=0)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("area.embed")(probe)
