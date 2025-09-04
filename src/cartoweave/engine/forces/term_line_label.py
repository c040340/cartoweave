from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.geometry import project_point_to_segment, rect_half_extent_along_dir
from cartoweave.utils.kernels import (
    softplus, sigmoid, softabs,
    invdist_energy, invdist_force_mag,
    EPS_DIST, EPS_NORM, EPS_ABS,
)
from cartoweave.utils.shape import as_nx2

@register("ln.rect")
def term_line_label(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}
    segs = scene.get("lines")
    if segs is None or len(segs) == 0:
        return 0.0, np.zeros_like(P), {}

    segs = np.asarray(segs, float).reshape(-1,4)
    N = P.shape[0]
    WH_raw = scene.get("WH")
    if WH_raw is None:
        WH_raw = np.ones((N, 2))
    WH = as_nx2(WH_raw, N, "WH")
    F    = np.zeros_like(P); E = 0.0

    k_out = float(cfg.get("ln.k.repulse", 0.0))
    k_in  = float(cfg.get("ln.k.inside",  0.0))
    pwr   = float(cfg.get("ln.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_a = float(cfg.get("eps.abs",  EPS_ABS))
    beta_sep = float(cfg.get("ln.beta.sep", 6.0))
    beta_in  = float(cfg.get("ln.beta.in",  6.0))
    g_eps    = float(cfg.get("ln.g_eps",    1e-6))

    for i in range(P.shape[0]):
        cx, cy = float(P[i,0]), float(P[i,1])
        w, h   = float(WH[i,0]), float(WH[i,1])
        for s in segs:
            ax, ay, bx, by = map(float, s)
            qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
            nx, ny = -ty, tx   # 线段左法向

            # 将问题转到矩形边法向/切向坐标上，仍做轴向分解的平滑外距
            dx, dy = (cx - qx), (cy - qy)

            # 沿法向的“半径”与签名外距（把 label 当矩形在 n 方向的半径）
            r_n = rect_half_extent_along_dir(w, h, nx, ny, eps_abs=eps_a)
            s_n = softabs(nx*dx + ny*dy, eps_a) - r_n  # 法向外距（≥? 可能<0）
            s_t = softabs(tx*dx + ty*dy, eps_a)        # 切向距离（用于外距合成）

            spn = softplus(s_n, beta_sep)
            spt = softplus(s_t, beta_sep)
            g   = (spn*spn + spt*spt + g_eps*g_eps) ** 0.5

            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = - invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_dC_n = (spn / g) * sigmoid(beta_sep * s_n)
                    d_g_dC_t = (spt / g) * sigmoid(beta_sep * s_t)
                    # ∂s_n/∂C ≈ sign-like → 用 softabs 导数：in n-dir 就是 n 分量
                    coeff_n = (nx*dx + ny*dy) / (softabs(nx*dx + ny*dy, eps_a) + 1e-12)
                    coeff_t = (tx*dx + ty*dy) / (softabs(tx*dx + ty*dy, eps_a) + 1e-12)
                    dC_x = d_g_dC_n * coeff_n * nx + d_g_dC_t * coeff_t * tx
                    dC_y = d_g_dC_n * coeff_n * ny + d_g_dC_t * coeff_t * ty
                    fx = - dEdg * dC_x
                    fy = - dEdg * dC_y
                else:
                    fx = fy = 0.0
                F[i,0] += fx; F[i,1] += fy

            if k_in > 0.0:
                vin = softplus(-s_n, beta_in)  # 只有越线（s_n<0）才会有
                E  += 0.5 * k_in * (vin*vin)
                coeff_n = (nx*dx + ny*dy) / (softabs(nx*dx + ny*dy, eps_a) + 1e-12)
                dE_dC_n = k_in * vin * (-sigmoid(-beta_in * s_n)) * coeff_n
                fx_in = - dE_dC_n * nx
                fy_in = - dE_dC_n * ny
                F[i,0] += fx_in; F[i,1] += fy_in

    return float(E), F, {"ln": int(segs.shape[0])}
