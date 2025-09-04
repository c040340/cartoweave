# src/cartoweave/engine/forces/term_area_cross.py
from __future__ import annotations
import math
import numpy as np
from . import register

from cartoweave.utils.kernels import (
    softplus, sigmoid, softabs,
    EPS_DIST, EPS_NORM, EPS_ABS,
)
from cartoweave.utils.geometry import (
    project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
)

def segment_intersects_rect(ax, ay, bx, by, cx, cy, w, h, pad=0.0) -> bool:
    # simple AABB overlap with rectangle expanded by pad
    x1, x2 = sorted([ax, bx]); y1, y2 = sorted([ay, by])
    rx1, rx2 = (cx - w*0.5 - pad), (cx + w*0.5 + pad)
    ry1, ry2 = (cy - h*0.5 - pad), (cy + h*0.5 + pad)
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

@register("area.cross")
def term_area_cross(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    labels = scene.get("labels", [])
    areas  = scene.get("areas", [])
    WH     = np.asarray(scene.get("WH", np.zeros_like(P)), float)

    k_cross   = float(cfg.get("area.k.cross", 900.0))
    min_gap   = float(cfg.get("area.cross.min_gap", 1.5))
    eta_tan   = float(cfg.get("area.cross.eta", 2.0))
    alpha_sp  = float(cfg.get("area.cross.alpha", 0.35))
    cap_scale = float(cfg.get("area.cross.tan_cap_scale", 1.0))
    use_lc    = bool(cfg.get("area.cross.use_logcosh", True))
    p0_lc     = float(cfg.get("area.cross.sat_p0", 2.0))
    g_min_int = float(cfg.get("area.cross.gate_min_interior", 0.6))

    eps_abs   = float(cfg.get("eps.abs", EPS_ABS))

    F = np.zeros_like(P, float)
    E = 0.0
    S = [[] for _ in range(P.shape[0])]

    for i, lab in enumerate(labels):
        own_idx = int(lab.get("anchor_index", -1)) if lab.get("anchor_kind") == "area" else -1

        w, h = float(WH[i,0]), float(WH[i,1])
        if w <= 0.0 and h <= 0.0:
            continue
        hx, hy = 0.5*w, 0.5*h
        cx, cy = float(P[i,0]), float(P[i,1])

        for ai, A in enumerate(areas):
            if ai == own_idx:
                continue
            poly = A.get("polygon", None)
            if poly is None:
                continue
            arr = np.asarray(poly, float).reshape(-1, 2)
            if arr.shape[0] < 3:
                continue
            ccw = (poly_signed_area(arr) > 0.0)

            fx_sum = fy_sum = 0.0
            best = 0.0
            n = arr.shape[0]

            for k in range(n):
                ax, ay = float(arr[k,0]), float(arr[k,1])
                bx, by = float(arr[(k+1)%n,0]), float(arr[(k+1)%n,1])

                qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
                nx_in, ny_in = (-ty, tx) if ccw else (ty, -tx)

                dx, dy = (cx - qx), (cy - qy)
                s  = nx_in*dx + ny_in*dy
                u  = tx*dx + ty*dy
                abs_u = softabs(u, eps_abs)
                abs_s = softabs(s, eps_abs)
                r_n = hx*softabs(nx_in, eps_abs) + hy*softabs(ny_in, eps_abs)

                # depth & gating
                p = softplus((r_n + min_gap) - abs_s, alpha_sp)
                u_cap = cap_scale * max(hx, hy)
                g = sigmoid(-(abs_u - u_cap) / max(eta_tan, 1e-9))

                if segment_intersects_rect(ax, ay, bx, by, cx, cy, w, h, pad=min_gap):
                    g = 1.0
                elif 0.0 < t < 1.0:
                    g = max(g, g_min_int)

                x_gp = g * p
                if use_lc:
                    denom = max(p0_lc, 1e-9)
                    t0 = x_gp / denom
                    E_k = k_cross * (denom * (abs(t0) + math.log1p(math.exp(-2.0*abs(t0))) - math.log(2.0)))
                    dEdx = k_cross * math.tanh(t0)
                else:
                    E_k  = 0.5 * k_cross * (x_gp*x_gp)
                    dEdx = k_cross * x_gp

                # d(g*p)/dC
                coeff_u = u / max(abs_u, 1e-9)
                gprime = g * (1.0 - g) * (-1.0 / max(eta_tan, 1e-9)) * coeff_u
                dgx, dgy = gprime * tx, gprime * ty

                coeff_s = s / max(abs_s, 1e-9)
                sig_az  = sigmoid(alpha_sp * ((r_n + min_gap) - abs_s))
                dpx, dpy = sig_az * (-coeff_s) * nx_in, sig_az * (-coeff_s) * ny_in

                dx_dc_x = p * dgx + g * dpx
                dx_dc_y = p * dgy + g * dpy

                fx_k = -(dEdx * dx_dc_x)
                fy_k = -(dEdx * dx_dc_y)

                fx_sum += fx_k
                fy_sum += fy_k
                best = max(best, abs(dEdx))
                E += E_k  # 能量要累加

            F[i,0] += fx_sum
            F[i,1] += fy_sum
            S[i].append((int(ai), float(fx_sum), float(fy_sum), float(best)))

    return float(E), F, {"area_cross": S}
