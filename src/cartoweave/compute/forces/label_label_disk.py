# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register
from ._common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    invdist_energy,
    invdist_force_mag,
)


@register("ll.disk")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("ll.disk", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.disk"}

    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids_solver", list(range(N)))
    assert len(active_ids) == N, f"_active_ids_solver misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    k_out = float(cfg.get("ll.k.repulse", 900.0))
    p = float(cfg.get("ll.edge_power", 2.0))
    edge_eps = float(cfg.get("ll.edge_eps", 0.5))
    beta = float(cfg.get("beta.softplus.sep", cfg.get("beta_softplus_sep", 6.0)))
    k_in = float(cfg.get("ll.k.inside", 0.0))
    mode = str(cfg.get("ll.disk.mode", "max"))

    def radius_from_wh(w, h):
        if mode == "min":
            return 0.5 * min(w, h)
        if mode == "avg":
            return 0.25 * (w + h)
        if mode == "diag":
            return 0.5 * math.hypot(w, h)
        return 0.5 * max(w, h)

    v0 = math.log(2.0) / max(beta, eps)
    e0 = v0 + edge_eps
    if k_in <= 0.0:
        k_in = k_out / ((e0 ** p) * max(v0, eps))

    F = np.zeros_like(P)
    E = 0.0
    src = [[] for _ in range(N)]

    for ii in range(len(idxs)):
        i = idxs[ii]
        wi, hi = float(WH[i, 0]), float(WH[i, 1])
        xi, yi = float(P[i, 0]), float(P[i, 1])
        if wi <= 0.0 and hi <= 0.0:
            continue
        ri = radius_from_wh(wi, hi)
        for jj in range(ii + 1, len(idxs)):
            j = idxs[jj]
            wj, hj = float(WH[j, 0]), float(WH[j, 1])
            xj, yj = float(P[j, 0]), float(P[j, 1])
            rj = radius_from_wh(wj, hj)
            dx, dy = xi - xj, yi - yj
            rc = math.hypot(dx, dy) + eps
            s = rc - (ri + rj)
            c = softplus(s, beta) + edge_eps
            v = softplus(-s, beta)
            sc = sigmoid(beta * s)
            sv = sigmoid(-beta * s)
            E += invdist_energy(c, k_out, p) + 0.5 * k_in * (v * v)
            fmag = invdist_force_mag(c, k_out, p) * sc + (k_in * v * sv)
            ux, uy = dx / rc, dy / rc
            fx, fy = fmag * ux, fmag * uy
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy
            src[i].append((int(j), float(fx), float(fy), float(abs(fmag))))
            src[j].append((int(i), float(-fx), float(-fy), float(abs(fmag))))

    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"source": "compute.forces.ll.disk", "ll.disk": src}
