# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from ._common import get_eps, weight_of, ensure_vec2

@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    """
    Return (E, F, meta) with F shape (L,2).
    No inactive zeroing here; aggregator will do it.
    """
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("anchor.spring", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    A = scene.get("anchors")
    if A is None:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    term_cfg = cfg.get("terms", {}).get("anchor", {}).get("spring", {})
    k = float(term_cfg.get("k", 10.0))
    r0 = float(term_cfg.get("r0", 0.0))

    N = P.shape[0]
    A = np.asarray(A, float)
    assert A.shape[0] == N, f"anchors misaligned: {A.shape} vs P {P.shape}"
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    ext_dir = scene.get("_ext_dir")
    if ext_dir is None:
        ext_dir = np.zeros((N, 2))
    else:
        ext_dir = np.asarray(ext_dir, float)
        assert ext_dir.shape[0] == N, f"ext_dir misaligned: {ext_dir.shape} vs P {P.shape}"

    F = np.zeros_like(P)
    E = 0.0
    for i in idxs:
        px, py = P[i]
        ax, ay = A[i]
        dx = px - ax
        dy = py - ay
        r = (dx * dx + dy * dy) ** 0.5
        if r > eps:
            u = np.array([dx, dy]) / (r + eps)
            coeff = -k * (r - r0)
            fx, fy = coeff * u
        else:
            u_ext = ext_dir[i]
            nrm = float(np.hypot(u_ext[0], u_ext[1]))
            if nrm > 0.0:
                u = u_ext / (nrm + eps)
            else:
                u = np.array([0.0, 0.0])
            coeff = -k * (0.0 - r0)
            fx, fy = coeff * u
        F[i, 0] += fx
        F[i, 1] += fy
        E += 0.5 * k * (r - r0) ** 2

    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"source": "compute.forces.anchor.spring"}
