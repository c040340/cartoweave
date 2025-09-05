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


@register("ll.rect")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("ll.rect", cfg, 0.0)
    if w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.rect"}

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

    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("ll.k.repulse", 0.0))
    k_in = float(cfg.get("ll.k.inside", 0.0))
    pwr = float(cfg.get("ll.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", eps))
    eps_n = float(cfg.get("eps.norm", eps))
    eps_a = float(cfg.get("eps.abs", 1e-3))
    beta_sep = float(cfg.get("ll.beta.sep", 6.0))
    beta_in = float(cfg.get("ll.beta.in", 6.0))
    g_eps = float(cfg.get("ll.g_eps", 1e-6))

    for ai in range(len(idxs)):
        a = idxs[ai]
        xa, ya = float(P[a, 0]), float(P[a, 1])
        wa, ha = float(WH[a, 0]), float(WH[a, 1])
        for bi in range(ai + 1, len(idxs)):
            b = idxs[bi]
            xb, yb = float(P[b, 0]), float(P[b, 1])
            wb, hb = float(WH[b, 0]), float(WH[b, 1])
            dx, dy = xa - xb, ya - yb
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * (wa + wb)
            sy = ady - 0.5 * (ha + hb)
            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5
            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    dspx_dsx = sigmoid(beta_sep * sx)
                    dsy_dy = None
                    d_adx_ddx = dx / (adx + eps)
                    d_ady_ddy = dy / (ady + eps)
                    d_g_ddx = (spx / g) * dspx_dsx * d_adx_ddx
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * d_ady_ddy
                    fx = -dEdg * d_g_ddx
                    fy = -dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[a, 0] += fx
                F[a, 1] += fy
                F[b, 0] -= fx
                F[b, 1] -= fy
            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                c1 = k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                c2 = k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                fx_in = -c1
                fy_in = -c2
                F[a, 0] += fx_in
                F[a, 1] += fy_in
                F[b, 0] -= fx_in
                F[b, 1] -= fy_in

    F = ensure_vec2(F, L)
    M = len(idxs)
    return float(E * w), F * w, {"source": "compute.forces.ll.rect", "pairs": int(M * (M - 1) // 2)}
