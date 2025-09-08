# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, term_cfg, eps_params
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


@register("focus.attract")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}
    N = int(P.shape[0])
    tc = term_cfg(cfg, "focus", "attract")
    epss = eps_params(cfg, tc, defaults={})
    eps = epss["eps_numeric"]

    k = float(0.8 if tc.get("k") is None else tc.get("k"))
    if k <= 0.0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    center = tc.get("center", None)
    if center is None:
        center = scene.get("focus_center", None)
    if center is None:
        fw, fh = scene.get("frame_size", (0.0, 0.0))
        center = np.array([float(fw) * 0.5, float(fh) * 0.5], dtype=float)
    Cx, Cy = float(center[0]), float(center[1])

    sigma = tc.get("sigma") or {}
    sigx = float(100.0 if sigma.get("x") is None else sigma.get("x"))
    sigy = float(100.0 if sigma.get("y") is None else sigma.get("y"))
    delta = float(8.0 if tc.get("delta") is None else tc.get("delta"))
    only_free = bool(tc.get("only_free") if tc.get("only_free") is not None else False)

    if sigx <= eps or sigy <= eps:
        sigx = max(sigx, eps)
        sigy = max(sigy, eps)

    labels = read_labels_aligned(scene, P)
    WH = normalize_WH_from_labels(labels, N, "focus.attract")
    modes = [get_mode(l) for l in labels]
    mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P, float)
    E = 0.0
    info = []

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        if only_free:
            a = _anchor(lab)
            kind = a["kind"] if a and a["kind"] is not None else "none"
            if kind and kind != "none":
                continue
        x, y = float(P[i, 0]), float(P[i, 1])
        rx = (x - Cx) / sigx
        ry = (y - Cy) / sigy
        Q = rx * rx + ry * ry
        root = (1.0 + Q / (delta * delta)) ** 0.5
        E_i = k * (delta * delta) * (root - 1.0)
        denom = max(root, eps)
        dEdx = k * (rx / (sigx * denom))
        dEdy = k * (ry / (sigy * denom))
        fx = -dEdx
        fy = -dEdy
        F[i, 0] += fx
        F[i, 1] += fy
        E += E_i
        info.append((int(i), float(E_i), float(fx), float(fy)))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "focus.attract", "focus_huber": info}
