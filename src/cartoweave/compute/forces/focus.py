# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register
from ._common import get_eps, weight_of, ensure_vec2


@register("focus.attract")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("focus.attract", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    k = float(cfg.get("focus.k.attract", 0.0))
    if k <= 0.0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    center = cfg.get("focus.center", None)
    if center is None:
        center = scene.get("focus_center", None)
    if center is None:
        fw, fh = scene.get("frame_size", (0.0, 0.0))
        center = np.array([float(fw) * 0.5, float(fh) * 0.5], dtype=float)
    Cx, Cy = float(center[0]), float(center[1])

    sigx = float(cfg.get("focus.sigma.x", 100.0))
    sigy = float(cfg.get("focus.sigma.y", 100.0))
    delta = float(cfg.get("focus.delta", 8.0))
    only_free = bool(cfg.get("focus.only_free", False))
    sigx = max(sigx, eps)
    sigy = max(sigy, eps)

    labels_all = scene.get("labels", [])
    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P, float)
    E = 0.0
    info = []
    for i in idxs:
        lab = labels[i]
        if only_free:
            kind = lab.get("anchor_kind", "none")
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

    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"source": "compute.forces.focus.attract", "focus_huber": info}
