# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from ._common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import softplus, sigmoid, softabs
from cartoweave.utils.geometry import (
    project_point_to_segment,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.numerics import sigmoid_np, d_sigmoid_np, softabs_np


@register("area.embed")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("area.embed", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.embed"}

    labels_all = scene.get("labels", [])
    areas = scene.get("areas", [])
    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    term_cfg = cfg.get("terms", {}).get("area_embed", {})
    k_embed = float(term_cfg.get("k", 0.8))
    k_tan = k_embed
    ratio_in = 0.60
    gate_eta = 2.0
    gate_slack = 1.0
    edge_bias = float(np.clip(term_cfg.get("edge_bias", 0.0), 0.0, 1.0))
    st_cfg = (
        cfg.get("solver", {})
        .get("internals", {})
        .get("stability", {})
    )
    sigma = max(float(term_cfg.get("sigma", 6.0)), float(st_cfg.get("eps_sigma", 1.0e-3)))
    exp_clip = float(st_cfg.get("exp_clip", 40.0))
    eps_norm = float(st_cfg.get("eps_norm", eps))
    area_eps = float(st_cfg.get("area_eps", eps))
    eps_abs = float(cfg.get("eps.abs", 1e-3))

    F = np.zeros_like(P, float)
    E_total = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        if lab.get("anchor_kind") != "area":
            continue
        ai = int(lab.get("anchor_index", -1))
        if ai < 0 or ai >= len(areas):
            continue
        poly = areas[ai].get("polygon", None)
        if poly is None:
            continue
        arr = np.asarray(poly, float).reshape(-1, 2)
        if not np.isfinite(arr).all():
            continue
        nE = arr.shape[0]
        if nE < 3:
            continue
        if abs(poly_signed_area(arr)) <= area_eps:
            continue
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
        coeff_s = s_list / np.maximum(sabs, eps)
        d2 = s_list * s_list
        expo = -d2 / (2.0 * sigma * sigma)
        expo = np.clip(expo, -exp_clip, exp_clip)
        w0 = np.exp(expo)
        dv_ds_w = s_list / (sigma * sigma)
        dv_dC = (dv_ds_w[:, None]) * n_list
        dw0_dC = w0[:, None] * (-dv_dC)
        if edge_bias != 0.0:
            w_raw = (1.0 - edge_bias) * w0 + edge_bias
            dw_raw_dC = (1.0 - edge_bias) * dw0_dC
        else:
            w_raw = w0
            dw_raw_dC = dw0_dC
        sum_w = float(np.sum(w_raw))
        sum_w = max(sum_w, eps_norm)
        wgt = w_raw / sum_w
        d_sum_w_dC = dw_raw_dC.sum(axis=0)
        dwgt_dC = (dw_raw_dC * sum_w - w_raw[:, None] * d_sum_w_dC) / (sum_w * sum_w)
        if not np.isfinite(wgt).all():
            continue

        s_star = (2.0 * ratio_in - 1.0) * rn_list
        ds = s_list - s_star
        E_perp = 0.5 * k_embed * (ds * ds)
        inv_eta = 1.0 / max(gate_eta, eps)
        x = ((rn_list + gate_slack) - sabs) * inv_eta
        g = sigmoid_np(x)
        if not np.isfinite(g).all():
            raise FloatingPointError("area_embed: non-finite gate")
        E_tan = 0.5 * k_tan * g * (u_list * u_list)
        E_k = E_perp + E_tan
        E = float((wgt * E_k).sum())
        E_total += E

        dEperp_dC = (k_embed * ds)[:, None] * n_list
        gprime = d_sigmoid_np(x) * (-inv_eta) * coeff_s
        dEtan_dC = (0.5 * k_tan * (u_list * u_list) * gprime)[:, None] * n_list \
                 + (k_tan * g * u_list)[:, None] * t_list
        dEk_dC = dEperp_dC + dEtan_dC
        grad = (wgt[:, None] * dEk_dC).sum(axis=0) + (E_k[:, None] * dwgt_dC).sum(axis=0)
        if not np.isfinite(grad).all():
            continue
        F[i, 0] += -grad[0]
        F[i, 1] += -grad[1]
        k_min = int(np.argmin(sabs))
        S[i].append((
            int(ai), float(-grad[0]), float(-grad[1]),
            float(E), k_min, float(s_list[k_min]), float(u_list[k_min]),
            float(wgt[k_min])
        ))

    F = ensure_vec2(F, L)
    return float(E_total * w), F * w, {"source": "compute.forces.area.embed", "area_embed": S}
