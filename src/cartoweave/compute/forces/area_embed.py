# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import (
    EPS_DIST,
    EPS_NORM,
    EPS_ABS,
)
from cartoweave.utils.geometry import (
    project_point_to_segment,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger
from cartoweave.utils.numerics import (
    sigmoid_np,
    d_sigmoid_np,
    softabs_np,
    softmin_weights_np,
)


@register("area.embed")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.embed"}
    L = P.shape[0]
    eps = get_eps(cfg)
    w = weight_of("area.embed", cfg, 0.0)
    if w <= 0.0:
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
    skip_circle = int(np.count_nonzero(~mask))

    k_embed = float(cfg.get("area.k.embed", 200.0))
    k_tan = float(cfg.get("area.k.tan", 30.0))
    ratio_in = float(cfg.get("area.embed.ratio_in", 0.60))
    gate_eta = float(cfg.get("area.tan.gate.eta", 2.0))
    gate_slack = float(cfg.get("area.tan.gate.slack", 1.0))
    beta_edge = float(cfg.get("area.embed.beta_edge", 6.0))
    eps_abs = float(cfg.get("eps.abs", EPS_ABS))
    eps_div = eps

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
        nE = arr.shape[0]
        if nE < 3:
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

    logger.debug("term_area_embed: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E_total * w), F * w, {"term": "area.embed", "area_embed": S}
