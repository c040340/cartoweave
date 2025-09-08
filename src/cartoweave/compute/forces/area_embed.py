# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps
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


@register("area.embed")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.embed"}
    eps = get_eps(cfg)

    labels = read_labels_aligned(scene, P)
    areas = scene.get("areas", [])
    N = int(P.shape[0])
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]
    WH = normalize_WH_from_labels(labels, N, "area.embed")

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
        a = _anchor(lab)
        if not a or a["kind"] != "area":
            continue
        ai = int(a["index"]) if a["index"] is not None else -1
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

    F = ensure_vec2(F, N)
    return float(E_total), F, {"term": "area.embed", "area_embed": S}
