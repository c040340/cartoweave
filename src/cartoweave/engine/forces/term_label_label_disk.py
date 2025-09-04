# src/cartoweave/engine/forces/term_label_label_disk.py
from __future__ import annotations
import math
import numpy as np
from . import register

from cartoweave.utils.kernels import (
    softplus, sigmoid, softabs,
    invdist_energy, invdist_force_mag,
    EPS_DIST, EPS_NORM, EPS_ABS, softmin_weights,
)
from cartoweave.utils.shape import as_nx2

from cartoweave.utils.geometry import (
    project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
)

@register("ll.disk")
def term_ll_disk(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    N = P.shape[0]
    WH_raw = scene.get("WH")
    if WH_raw is None:
        WH_raw = np.ones((N, 2))
    WH = as_nx2(WH_raw, N, "WH")

    k_out  = float(cfg.get("ll.k.repulse", 900.0))
    p      = float(cfg.get("ll.edge_power", 2.0))
    eps    = float(cfg.get("ll.edge_eps", 0.5))
    beta   = float(cfg.get("beta.softplus.sep", cfg.get("beta_softplus_sep", 6.0)))
    k_in   = float(cfg.get("ll.k.inside", 0.0))  # if 0 → auto continuous matching
    mode   = str(cfg.get("ll.disk.mode", "max"))

    def radius_from_wh(w: float, h: float) -> float:
        if mode == "min":  return 0.5 * min(w, h)
        if mode == "avg":  return 0.25 * (w + h)
        if mode == "diag": return 0.5 * math.hypot(w, h)
        return 0.5 * max(w, h)

    # match inside coefficient at s=0
    v0 = math.log(2.0) / max(beta, 1e-8)
    e0 = v0 + eps
    if k_in <= 0.0:
        k_in = k_out / ((e0 ** p) * max(v0, 1e-8))

    F = np.zeros_like(P)
    E = 0.0
    src = [[] for _ in range(N)]

    for i in range(N):
        wi, hi = float(WH[i,0]), float(WH[i,1])
        xi, yi = float(P[i,0]), float(P[i,1])
        if wi <= 0.0 and hi <= 0.0:
            continue
        ri = radius_from_wh(wi, hi)
        for j in range(i+1, N):
            wj, hj = float(WH[j,0]), float(WH[j,1])
            xj, yj = float(P[j,0]), float(P[j,1])
            rj = radius_from_wh(wj, hj)

            dx, dy = xi - xj, yi - yj
            rc = math.hypot(dx, dy) + 1e-9  # center distance
            s  = rc - (ri + rj)

            # outside barrier + inside quad (softplus gating)
            c  = softplus(s, beta) + eps
            v  = softplus(-s, beta)
            sc = sigmoid(beta * s)
            sv = sigmoid(-beta * s)

            E += invdist_energy(c, k_out, p) + 0.5 * k_in * (v*v)

            fmag = invdist_force_mag(c, k_out, p) * sc + (k_in * v * sv)
            ux, uy = dx/rc, dy/rc
            fx, fy = fmag*ux, fmag*uy

            F[i,0] += fx;  F[i,1] += fy
            F[j,0] -= fx;  F[j,1] -= fy

            src[i].append((int(j), float(fx), float(fy), float(abs(fmag))))
            src[j].append((int(i), float(-fx), float(-fy), float(abs(fmag))))

    return float(E), F, {"ll.disk": src}
