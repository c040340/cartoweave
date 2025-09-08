# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg, eps_params
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
    return {"kind": getattr(a, "kind", None), "index": getattr(a, "index", None), "t": getattr(a, "t", None)}


def _focus_force_at(xy: np.ndarray, center: np.ndarray, sigma_xy: np.ndarray, delta: float, k: float) -> np.ndarray:
    """Closed-form focus attraction force at ``xy`` points."""

    xy = np.asarray(xy, dtype=float)
    rx = (xy[:, 0] - center[0]) / sigma_xy[0]
    ry = (xy[:, 1] - center[1]) / sigma_xy[1]
    Q = rx * rx + ry * ry
    root = np.sqrt(1.0 + Q / (delta * delta))
    denom = np.maximum(root, 1e-12)
    Fx = -k * rx / (sigma_xy[0] * denom)
    Fy = -k * ry / (sigma_xy[1] * denom)
    return np.stack([Fx, Fy], axis=1)


@register("focus.attract")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}
    N = int(P.shape[0])
    tc = term_cfg(cfg, "focus", "attract")
    epss = eps_params(cfg, tc, defaults={})
    eps = epss["eps_numeric"]

    k_attract = float(0.8 if tc.get("k_attract") is None else tc.get("k_attract"))
    if k_attract <= 0.0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    center = tc.get("center", None)
    if center is None:
        center = scene.get("focus_center", None)
    if center is None:
        fw, fh = scene.get("frame_size", (0.0, 0.0))
        center = np.array([float(fw) * 0.5, float(fh) * 0.5], dtype=float)
    W, H = scene.get("frame_size", (1920.0, 1080.0))
    W, H = float(W), float(H)
    Cx, Cy = float(center[0]) * W, float(center[1]) * H
    sigma = float(tc.get("sigma") or 1.0)
    wh = tc.get("wh")
    if wh is not None:
        wx, hy = float(wh[0]), float(wh[1])
        sigx = max(eps, sigma * wx * W)
        sigy = max(eps, sigma * hy * H)
    else:
        sigx = sigy = max(sigma, eps)
    delta = float(8.0 if tc.get("delta") is None else tc.get("delta"))
    only_free = bool(tc.get("only_free") if tc.get("only_free") is not None else False)

    labels = read_labels_aligned(scene, P)
    WH = normalize_WH_from_labels(labels, N, "focus.attract")
    modes = [get_mode(l) for l in labels]
    mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P, float)
    E = 0.0
    info = []

    center_xy = np.array([Cx, Cy], float)
    sig_xy = np.array([sigx, sigy], float)
    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        if only_free:
            a = _anchor(lab)
            kind = a["kind"] if a and a["kind"] is not None else "none"
            if kind and kind != "none":
                continue
        x, y = float(P[i, 0]), float(P[i, 1])
        Fi = _focus_force_at(np.array([[x, y]], float), center_xy, sig_xy, delta, k_attract)[0]
        rx = (x - Cx) / sigx
        ry = (y - Cy) / sigy
        Q = rx * rx + ry * ry
        root = (1.0 + Q / (delta * delta)) ** 0.5
        E_i = k_attract * (delta * delta) * (root - 1.0)
        fx, fy = float(Fi[0]), float(Fi[1])
        F[i, 0] += fx
        F[i, 1] += fy
        E += E_i
        info.append((int(i), float(E_i), float(fx), float(fy)))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "focus.attract", "focus_huber": info}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample focus attraction field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    frame_size = scene.get("frame_size", (1000.0, 1000.0))
    W, H = float(frame_size[0]), float(frame_size[1])

    center = params.get("center")
    if center is None:
        center = scene.get("focus_center")
    if center is None:
        center = (0.5, 0.5)
    Cx, Cy = float(center[0]) * W, float(center[1]) * H

    wh = params.get("wh") or [0.20, 0.15]
    sigx = wh[0] * W
    sigy = wh[1] * H
    sigma = float(8.0 if params.get("sigma") is None else params.get("sigma"))
    k = float(0.8 if params.get("k_attract") is None else params.get("k_attract"))

    F = _focus_force_at(xy, np.array([Cx, Cy], float), np.array([sigx, sigy], float), sigma, k)
    if not np.isfinite(F).all():
        raise ValueError("focus.probe produced non-finite values")
    return F


register_probe("focus.attract")(probe)
