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

@register("ll.rect")
def term_ll_rect(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    N = P.shape[0]
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

    F  = np.zeros_like(P)
    E  = 0.0

    k_out = float(cfg.get("ll.k.repulse", 0.0))
    k_in  = float(cfg.get("ll.k.inside",  0.0))
    pwr   = float(cfg.get("ll.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_n = float(cfg.get("eps.norm", EPS_NORM))
    eps_a = float(cfg.get("eps.abs",  EPS_ABS))

    beta_sep = float(cfg.get("ll.beta.sep", 6.0))
    beta_in  = float(cfg.get("ll.beta.in",  6.0))
    g_eps    = float(cfg.get("ll.g_eps",    1e-6))

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
            sx = adx - 0.5*(wa + wb)
            sy = ady - 0.5*(ha + hb)

            spx = softplus(sx, beta_sep)   # ≥0
            spy = softplus(sy, beta_sep)
            g   = (spx*spx + spy*spy + g_eps*g_eps) ** 0.5

            # ---------- 外距能量 ----------
            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                # dE/dg = -k_out * (g+eps)^(-pwr)
                dEdg = - invdist_force_mag(g + eps_d, k_out, pwr)

                # ∂g/∂dx = (spx / g) * d spx/dx
                if g > 0.0:
                    dspx_dsx = sigmoid(beta_sep * sx)
                    dsy_dy   = None  # not needed explicitly
                    d_adx_ddx = dx / (adx + 1e-12)     # d softabs / ddx
                    d_ady_ddy = dy / (ady + 1e-12)

                    d_g_ddx = (spx / g) * dspx_dsx * d_adx_ddx
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * d_ady_ddy

                    fx = - dEdg * d_g_ddx
                    fy = - dEdg * d_g_ddy
                else:
                    fx = fy = 0.0

                F[a,0] += fx; F[a,1] += fy
                F[b,0] -= fx; F[b,1] -= fy

            # ---------- 侵入能量（两轴都可能<0） ----------
            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)  # ≥0
                E  += 0.5 * k_in * (vin*vin)
                # dE/dC = k_in * vin * d(vin)/dC
                # d(softplus(-sx))/ddx = -sigmoid(-β sx) * d sx/ddx
                c1 = k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + 1e-12))
                c2 = k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + 1e-12))
                # ∂sx/∂dx = d|dx|/ddx = dx/|dx|_soft ; ∂sy/∂dy 同理
                fx_in = - c1
                fy_in = - c2
                F[a,0] += fx_in; F[a,1] += fy_in
                F[b,0] -= fx_in; F[b,1] -= fy_in

    logger.debug("term_ll_rect: skip_circle=%d", skip_circle)
    M = len(idxs)
    return float(E), F, {"pairs": int(M*(M-1)//2)}
