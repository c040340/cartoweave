# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from . import register, register_probe, term_cfg
from cartoweave.utils.geometry import polylines_to_segments
from cartoweave.utils.kernels import (
    invdist_energy,
    invdist_force_mag,
    sigmoid,
    softplus,
)
from ._common import (
    active_element_indices,
    ensure_vec2,
    float_param,
    get_mode,
    normalize_WH_from_labels,
    read_labels_aligned,
)


def softclamp(x: float, lo: float, hi: float, beta: float) -> float:
    """C¹ soft clamp of ``x`` to ``[lo, hi]``."""
    return lo + softplus(x - lo, beta) - softplus(x - hi, beta)


def d_softclamp(x: float, lo: float, hi: float, beta: float) -> float:
    """Derivative of :func:`softclamp` with respect to ``x``."""
    return sigmoid(beta * (x - lo)) - sigmoid(beta * (x - hi))


def point_rect_sdf_and_normal(px, py, cx, cy, w, h):
    """Signed distance and unit normal from rectangle center ``(cx,cy)`` to point ``(px,py)``."""
    dx = abs(px - cx) - 0.5 * w
    dy = abs(py - cy) - 0.5 * h
    ax = max(dx, 0.0)
    ay = max(dy, 0.0)
    if dx <= 0.0 and dy <= 0.0:
        sdf = -math.hypot(dx, dy)
        nx, ny = px - cx, py - cy
    else:
        sdf = math.hypot(ax, ay)
        ox = ax if px >= cx else -ax
        oy = ay if py >= cy else -ay
        nx, ny = ox, oy
    nlen = math.hypot(nx, ny)
    if nlen > 1e-9:
        nx /= nlen
        ny /= nlen
    else:
        nx, ny = 1.0, 0.0
    return float(sdf), np.array([nx, ny], float)


def soft_project_to_segment(cx, cy, ax, ay, bx, by, beta):
    """Softly project (cx,cy) onto segment AB and return point and Jacobian."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-18:
        return ax, ay, 0.0, 0.0, 0.0, 0.0, 0.0
    t_raw = ((cx - ax) * vx + (cy - ay) * vy) / L2
    t = softclamp(t_raw, 0.0, 1.0, beta)
    dt = d_softclamp(t_raw, 0.0, 1.0, beta)
    dt_dcx = (vx / L2) * dt
    dt_dcy = (vy / L2) * dt
    qx = ax + t * vx
    qy = ay + t * vy
    dqx_dcx = vx * dt_dcx
    dqx_dcy = vx * dt_dcy
    dqy_dcx = vy * dt_dcx
    dqy_dcy = vy * dt_dcy
    return qx, qy, dqx_dcx, dqx_dcy, dqy_dcx, dqy_dcy, math.sqrt(L2)


@register("ln.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "ln", "rect")
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}

    labels = read_labels_aligned(scene, P)
    N = int(P.shape[0])
    lines_all = scene.get("lines") or []
    modes = [get_mode(l) for l in labels]
    mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)

    active_ids = scene.get("_active_ids_solver")
    if active_ids is not None:
        active_mask = np.zeros(N, dtype=bool)
        active_mask[np.asarray(active_ids, dtype=int)] = True
        mask &= active_mask

    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}

    active_lines = active_element_indices([labels[i] for i in idxs], "line")
    lines = [lines_all[i] for i in sorted(active_lines) if 0 <= i < len(lines_all)]
    if not lines:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}

    WH = normalize_WH_from_labels(labels, N, "ln.rect")
    F = np.zeros_like(P)
    E = 0.0

    k_line = float(tc.get("k_line_sample", 100.0))
    p_exp = float(tc.get("line_sample_power", 2.0))
    min_gap = float(tc.get("line_sample_min_gap_px", 0.0))
    beta_d = float(tc.get("beta_softplus_dist", 3.0))
    eps_div = float(tc.get("eps_div", 1e-9))
    beta_t = float(tc.get("line_softproj_beta", 10.0))
    step_px = float(tc.get("line_sample_step_px", 12.0))

    for i in idxs:
        w, h = float(WH[i, 0]), float(WH[i, 1])
        if w <= 0.0 and h <= 0.0:
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for L in lines:
            pl = np.asarray(L, float).reshape(-1, 2)
            if pl.shape[0] < 2:
                continue
            seg = pl[1:] - pl[:-1]
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
            total = float(seg_len.sum())
            if not np.isfinite(total) or total <= 0.0:
                continue
            for k in range(seg.shape[0]):
                ax, ay = float(pl[k, 0]), float(pl[k, 1])
                bx, by = float(pl[k + 1, 0]), float(pl[k + 1, 1])
                qx, qy, dqx_dcx, dqx_dcy, dqy_dcx, dqy_dcy, Lk = soft_project_to_segment(
                    cx, cy, ax, ay, bx, by, beta_t
                )
                w_len = seg_len[k] / max(total, 1e-9)
                n_equiv = max(int(seg_len[k] / max(step_px, 1e-6)) + 1, 1)
                w_scale = w_len / n_equiv
                sd, n = point_rect_sdf_and_normal(qx, qy, cx, cy, w, h)
                d_raw = sd - min_gap
                d_eff = softplus(d_raw, beta_d) + eps_div
                sdr = sigmoid(beta_d * d_raw)
                E += w_scale * invdist_energy(d_eff, k_line, p_exp)
                fmag = w_scale * invdist_force_mag(d_eff, k_line, p_exp) * sdr
                ds_dcx = n[0] * (dqx_dcx - 1.0) + n[1] * dqy_dcx
                ds_dcy = n[0] * dqx_dcy + n[1] * (dqy_dcy - 1.0)
                F[i, 0] += fmag * ds_dcx
                F[i, 1] += fmag * ds_dcy

    F = ensure_vec2(F, N)
    seg_count = sum(
        max(np.asarray(pl, float).reshape(-1, 2).shape[0] - 1, 0) for pl in lines
    )
    return float(E), F, {"term": "ln.rect", "ln": int(seg_count)}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``ln.rect`` field from line segments acting on probes."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    segs_raw = scene.get("lines") or []
    if len(segs_raw) == 0:
        return np.zeros_like(xy, float)

    segs_arr = polylines_to_segments(segs_raw)
    segs = segs_arr.reshape((segs_arr.shape[0], 4))
    k_out = float_param(params, "k_out", 0.8)

    F = np.zeros_like(xy, float)
    for ax, ay, bx, by in segs:
        ax = float(ax)
        ay = float(ay)
        bx = float(bx)
        by = float(by)
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey) + 1e-9
        tx, ty = ex / L, ey / L
        vx = xy[:, 0] - ax
        vy = xy[:, 1] - ay
        t = np.clip(vx * tx + vy * ty, 0.0, L)
        qx = ax + t * tx
        qy = ay + t * ty
        dx = xy[:, 0] - qx
        dy = xy[:, 1] - qy
        dist_sq = dx * dx + dy * dy + 1e-9
        dist = np.sqrt(dist_sq)
        mag = k_out / dist_sq
        F[:, 0] += mag * dx / dist
        F[:, 1] += mag * dy / dist

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("ln.rect")(probe)

