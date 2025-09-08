# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    softclip,
    EPS_DIST,
    EPS_NORM,
    EPS_ABS,
)
from cartoweave.utils.geometry import (
    project_point_to_segment,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.numerics import softmin_weights_np
from cartoweave.utils.shape import as_nx2
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


@register("area.softout")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "area.softout"}

    labels = read_labels_aligned(scene, P)
    areas = scene.get("areas", []) or []
    N = int(P.shape[0])
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "area.softout")

    k_push = float(cfg.get("area.k.softout", 250.0))
    min_gap = float(params.get("min_gap", cfg.get("area.softout.min_gap", 0.0)))
    beta = float(cfg.get("area.softout.beta", 0.7))
    alpha_sp = float(cfg.get("area.softout.alpha", 0.35))
    gamma_out = float(cfg.get("area.softout.outside_weight", 0.5))
    lambda_out = float(cfg.get("area.softout.in_decay", 0.10))
    lambda_in = float(cfg.get("area.softout.out_decay", 0.06))
    eps_abs = float(cfg.get("eps.abs", EPS_ABS))

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        a = _anchor(lab)
        own_idx = int(a["index"]) if a and a["kind"] == "area" and a["index"] is not None else -1
        if w_i <= 0.0 and h_i <= 0.0:
            continue
        hx, hy = 0.5 * w_i, 0.5 * h_i
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for ai, poly in enumerate(areas):
            if ai == own_idx:
                continue
            # poly = A.get("polygon", None)
            if poly is None:
                continue
            arr = np.asarray(poly, float).reshape(-1, 2)
            if arr.shape[0] < 3:
                continue
            m_list = []
            n_list = []
            n = arr.shape[0]
            for k in range(n):
                ax, ay = float(arr[k, 0]), float(arr[k, 1])
                bx, by = float(arr[(k + 1) % n, 0]), float(arr[(k + 1) % n, 1])
                qx, qy, _, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
                ccw = (poly_signed_area(arr) > 0.0)
                nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)
                dx, dy = (cx - qx), (cy - qy)
                s_k = nx_in * dx + ny_in * dy
                r_n = hx * softabs(nx_in, eps_abs) + hy * softabs(ny_in, eps_abs)
                m_k = s_k - r_n - min_gap
                m_list.append(m_k)
                n_list.append((nx_in, ny_in))
            if not m_list:
                continue
            v = np.asarray(m_list, float)
            wts = softmin_weights_np(v, beta)
            m_eff = float((wts * v).sum())
            nx_eff = float((wts * np.asarray([n[0] for n in n_list])).sum())
            ny_eff = float((wts * np.asarray([n[1] for n in n_list])).sum())
            nrm = math.hypot(nx_eff, ny_eff)
            if nrm <= eps:
                continue
            nx_eff /= nrm
            ny_eff /= nrm
            if m_eff >= 0.0:
                r_in = softplus(m_eff, alpha_sp)
                s_in = sigmoid(alpha_sp * m_eff)
                # Hard clip replaced with softclip (C¹), keeps legacy rails (~[-80,80])
                t_in = softclip(lambda_in * m_eff, -80.0, 80.0, beta=8.0)
                decay_in = math.exp(-t_in)
                fmag = k_push * r_in * s_in * (decay_in ** 2)
                E += 0.5 * k_push * (r_in * decay_in) * (r_in * decay_in)
                mag = r_in * decay_in
            else:
                t_out = softclip(lambda_out * m_eff, -80.0, 80.0, beta=8.0)
                decay_out = math.exp(t_out)
                fmag = k_push * (gamma_out ** 2) * lambda_out * (decay_out ** 2)
                E += 0.5 * k_push * (gamma_out * decay_out) * (gamma_out * decay_out)
                mag = gamma_out * decay_out
            fx = fmag * (-nx_eff)
            fy = fmag * (-ny_eff)
            F[i, 0] += fx
            F[i, 1] += fy
            S[i].append((int(ai), float(fx), float(fy), float(mag)))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "area.softout", "area_softout": S}
