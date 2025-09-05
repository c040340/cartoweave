# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    invdist_energy,
    invdist_force_mag,
    EPS_DIST,
    EPS_NORM,
    EPS_ABS,
    softmin_weights,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("anchor.spring", cfg, 0.0)
    if phase != "anchor" or w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    A = scene.get("anchors")
    if A is None:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    k = float(cfg.get("anchor.k.spring", 10.0))
    r0_points = float(cfg.get("anchor.r0.points", 0.0))
    alpha = float(cfg.get("anchor.spring.alpha", 1.0))
    eps_n = float(cfg.get("eps.norm", EPS_NORM))

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
    skip_circle = int(np.count_nonzero(~mask))

    F = np.zeros_like(P)
    E = 0.0
    ext_dir = scene.get("_ext_dir")
    if ext_dir is None:
        ext_dir = np.zeros((N, 2))
    else:
        ext_dir = np.asarray(ext_dir, float)
        assert ext_dir.shape[0] == N, f"ext_dir misaligned: {ext_dir.shape} vs P {P.shape}"

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        px, py = P[i]
        ax, ay = A[i]
        dx = px - ax
        dy = py - ay
        r = (dx * dx + dy * dy) ** 0.5
        r0 = r0_points
        if r > eps:
            u = np.array([dx, dy]) / (r + eps_n)
            coeff = -k * (r - r0)
            fx, fy = coeff * u
        else:
            u_ext = ext_dir[i]
            nrm = float(np.hypot(u_ext[0], u_ext[1]))
            if nrm > 0:
                u = u_ext / (nrm + eps_n)
            else:
                u = np.array([0.0, 0.0])
            coeff = -k * (0.0 - r0)
            fx, fy = coeff * u
        F[i, 0] += fx
        F[i, 1] += fy
        E += 0.5 * k * (r - r0) ** 2

    logger.debug("term_anchor: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"term": "anchor.spring", "n": int(N)}
