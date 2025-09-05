# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from ._common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    invdist_energy,
    invdist_force_mag,
)


@register("pl.rect")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("pl.rect", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}
    pts = scene.get("points")
    if pts is None or len(pts) == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}

    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    pts = np.asarray(pts, float).reshape(-1, 2)
    M = pts.shape[0]
    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("pl.k.repulse", 0.0))
    k_in = float(cfg.get("pl.k.inside", 0.0))
    pwr = float(cfg.get("pl.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", eps))
    eps_a = float(cfg.get("eps.abs", 1e-3))
    beta_sep = float(cfg.get("pl.beta.sep", 6.0))
    beta_in = float(cfg.get("pl.beta.in", 6.0))
    g_eps = float(cfg.get("pl.g_eps", 1e-6))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for j in range(M):
            x, y = float(pts[j, 0]), float(pts[j, 1])
            dx, dy = cx - x, cy - y
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * w_i
            sy = ady - 0.5 * h_i
            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5
            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_ddx = (spx / g) * sigmoid(beta_sep * sx) * (dx / (adx + eps))
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * (dy / (ady + eps))
                    fx = -dEdg * d_g_ddx
                    fy = -dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[i, 0] += fx
                F[i, 1] += fy
            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                fx_in = -k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                fy_in = -k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                F[i, 0] += fx_in
                F[i, 1] += fy_in

    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"source": "compute.forces.pl.rect", "pl": int(N * M)}
