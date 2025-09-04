from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    invdist_energy,
    invdist_force_mag,
    EPS_DIST,
    EPS_NORM,
    EPS_ABS,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger

@register("pl.rect")
def term_point_label(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}
    pts = scene.get("points")
    if pts is None or len(pts) == 0:
        return 0.0, np.zeros_like(P), {}

    N = P.shape[0]
    WH_raw = scene.get("WH")
    if WH_raw is None:
        WH_raw = np.ones((N, 2))
    WH = as_nx2(WH_raw, N, "WH")
    pts = np.asarray(pts, float).reshape(-1,2)
    N, M = P.shape[0], pts.shape[0]
    labels = scene.get("labels", [])
    skip_hidden = 0
    skip_circle = 0
    F = np.zeros_like(P); E = 0.0

    k_out = float(cfg.get("pl.k.repulse", 0.0))
    k_in  = float(cfg.get("pl.k.inside",  0.0))
    pwr   = float(cfg.get("pl.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_a = float(cfg.get("eps.abs",  EPS_ABS))

    beta_sep = float(cfg.get("pl.beta.sep", 6.0))
    beta_in  = float(cfg.get("pl.beta.in",  6.0))
    g_eps    = float(cfg.get("pl.g_eps",    1e-6))

    for i in range(N):
        lab = labels[i] if i < len(labels) else {}
        w, h = float(WH[i, 0]), float(WH[i, 1])
        if lab.get("hidden"):
            assert w <= 0.0 and h <= 0.0
            skip_hidden += 1
            continue
        if lab.get("mode") == "circle":
            assert abs(w - h) < 1e-9
            skip_circle += 1
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for j in range(M):
            x, y = float(pts[j,0]), float(pts[j,1])
            dx, dy = cx - x, cy - y
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5*w
            sy = ady - 0.5*h

            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g   = (spx*spx + spy*spy + g_eps*g_eps) ** 0.5

            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = - invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_ddx = (spx / g) * sigmoid(beta_sep * sx) * (dx / (adx + 1e-12))
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * (dy / (ady + 1e-12))
                    fx = - dEdg * d_g_ddx
                    fy = - dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[i,0] += fx; F[i,1] += fy

            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E  += 0.5 * k_in * (vin*vin)
                fx_in = - k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + 1e-12))
                fy_in = - k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + 1e-12))
                F[i,0] += fx_in; F[i,1] += fy_in

    logger.debug(
        "term_point_label: skip_hidden=%d skip_circle=%d", skip_hidden, skip_circle
    )
    return float(E), F, {"pl": int(N*M)}
