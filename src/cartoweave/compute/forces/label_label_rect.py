# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    softabs,
    invdist_energy,
    invdist_force_mag,
)
from cartoweave.utils.shape import as_nx2  # noqa: F401
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
)


@register("ll.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    N = int(P.shape[0])
    tc = term_cfg(cfg, "ll", "rect")
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.rect"}

    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]

    # When the disk kernel is disabled we want all labels to participate in the
    # rectangle force irrespective of their declared ``ll_kernel``.  If both
    # kernels are enabled, respect the per‑label selection so that the two
    # implementations can coexist.
    disk_cfg = term_cfg(cfg, "ll", "disk")
    disk_enabled = bool(disk_cfg.get("enable", False))
    if disk_enabled:
        llk = [get_ll_kernel(l) for l in labels]
        rect_mask = np.array([(k or "") == "rect" for k in llk], dtype=bool)
    else:
        rect_mask = np.ones(N, dtype=bool)

    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask & rect_mask

    active_ids = scene.get("_active_ids_solver")
    if active_ids is not None:
        active_mask = np.zeros(N, dtype=bool)
        active_mask[np.asarray(active_ids, dtype=int)] = True
        mask &= active_mask

    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "ll.rect")

    F = np.zeros_like(P)
    E = 0.0

    k_out = float(tc.get("k_ll_repulse", 900.0))
    pwr = float(tc.get("ll_edge_power", 2.0))
    eps_edge = float(tc.get("ll_edge_eps", 0.5))
    beta_dir = float(tc.get("beta_softabs_dir", 6.0))
    beta_sep = float(tc.get("beta_softplus_sep", 6.0))
    v0 = math.log(2.0) / max(beta_sep, 1e-8)
    e0 = v0 + eps_edge
    k_in_auto = k_out / ((e0 ** pwr) * max(v0, 1e-8))
    k_in_val = tc.get("k_ll_inside")
    if k_in_val is None:
        k_in_val = k_in_auto
    k_in = float(k_in_val)
    inv_beta_dir = 1.0 / max(beta_dir, 1e-8)

    src = [[] for _ in range(N)]

    for ai in range(len(idxs)):
        a = idxs[ai]
        xa, ya = float(P[a, 0]), float(P[a, 1])
        wa, ha = float(WH[a, 0]), float(WH[a, 1])
        for bi in range(ai + 1, len(idxs)):
            b = idxs[bi]
            xb, yb = float(P[b, 0]), float(P[b, 1])
            wb, hb = float(WH[b, 0]), float(WH[b, 1])

            dx, dy = xa - xb, ya - yb
            hx = 0.5 * (wa + wb)
            hy = 0.5 * (ha + hb)

            rc = math.sqrt(dx * dx + dy * dy + inv_beta_dir * inv_beta_dir)
            ux = dx / rc
            uy = dy / rc

            ax = softabs(dx, inv_beta_dir) - hx
            ay = softabs(dy, inv_beta_dir) - hy

            px = softplus(ax, beta_sep)
            py = softplus(ay, beta_sep)
            d_out = math.hypot(px, py)
            d_eff = d_out + eps_edge

            nx = softplus(-ax, beta_sep)
            gx = sigmoid(-beta_sep * ax)
            ny = softplus(-ay, beta_sep)
            gy = sigmoid(-beta_sep * ay)
            mx = nx * gx
            my = ny * gy
            m_in = math.hypot(mx, my)

            E += invdist_energy(d_eff, k_out, pwr) + 0.5 * k_in * (m_in * m_in)

            dabsx = dx / max(math.sqrt(dx * dx + inv_beta_dir * inv_beta_dir), 1e-12)
            dabsy = dy / max(math.sqrt(dy * dy + inv_beta_dir * inv_beta_dir), 1e-12)
            if d_out > 0.0:
                dd_dpx = px / d_out
                dd_dpy = py / d_out
            else:
                dd_dpx = dd_dpy = 0.0
            dpx_dax = sigmoid(beta_sep * ax)
            dpy_day = sigmoid(beta_sep * ay)

            dd_ddx = dd_dpx * dpx_dax * dabsx
            dd_ddy = dd_dpy * dpy_day * dabsy
            fx_out = invdist_force_mag(d_eff, k_out, pwr) * dd_ddx
            fy_out = invdist_force_mag(d_eff, k_out, pwr) * dd_ddy

            if m_in > 0.0:
                dm_dmx = mx / m_in
                dm_dmy = my / m_in
            else:
                dm_dmx = dm_dmy = 0.0

            dnx_dax = -gx
            dgx_dax = gx * (1.0 - gx) * (-beta_sep)
            dmx_dax = dnx_dax * gx + nx * dgx_dax
            dax_ddx = dabsx

            dny_day = -gy
            dgy_day = gy * (1.0 - gy) * (-beta_sep)
            dmy_day = dny_day * gy + ny * dgy_day
            day_ddy = dabsy

            dm_ddx = dm_dmx * dmx_dax * dax_ddx
            dm_ddy = dm_dmy * dmy_day * day_ddy
            fx_in = -k_in * m_in * dm_ddx
            fy_in = -k_in * m_in * dm_ddy

            fx = fx_out + fx_in
            fy = fy_out + fy_in

            F[a, 0] += fx
            F[a, 1] += fy
            F[b, 0] -= fx
            F[b, 1] -= fy

            fmag = math.hypot(fx, fy)
            src[a].append((int(b), float(fx), float(fy), float(fmag), float(min(d_eff, m_in + eps_edge))))
            src[b].append((int(a), float(-fx), float(-fy), float(fmag), float(min(d_eff, m_in + eps_edge))))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "ll.rect", "ll.rect": src}


def _pairwise_force_rect(src_xy: np.ndarray, src_wh: np.ndarray, xy: np.ndarray, params: dict) -> np.ndarray:
    """Force from a rectangular source at ``src_xy`` acting on probe points ``xy``."""

    k_out = float_param(params, "k_ll_repulse", 900.0)
    pwr = float_param(params, "ll_edge_power", 2.0)
    eps_edge = float_param(params, "ll_edge_eps", 0.5)
    beta_dir = float_param(params, "beta_softabs_dir", 6.0)
    beta_sep = float_param(params, "beta_softplus_sep", 6.0)
    k_in = float_param(params, "k_ll_inside", float("nan"))
    v0 = math.log(2.0) / max(beta_sep, 1e-8)
    e0 = v0 + eps_edge
    if not np.isfinite(k_in) or k_in <= 0.0:
        k_in = k_out / ((e0 ** pwr) * max(v0, 1e-8))
    inv_beta_dir = 1.0 / max(beta_dir, 1e-8)

    dx = xy[:, 0] - src_xy[0]
    dy = xy[:, 1] - src_xy[1]
    hx = 0.5 * src_wh[0]
    hy = 0.5 * src_wh[1]

    rc = np.sqrt(dx * dx + dy * dy + inv_beta_dir * inv_beta_dir)
    ux = dx / rc
    uy = dy / rc

    ax = np.sqrt(dx * dx + inv_beta_dir * inv_beta_dir) - hx
    ay = np.sqrt(dy * dy + inv_beta_dir * inv_beta_dir) - hy

    px = softplus(ax, beta_sep)
    py = softplus(ay, beta_sep)
    d_out = np.hypot(px, py)
    d_eff = d_out + eps_edge

    nx = softplus(-ax, beta_sep)
    gx = sigmoid(-beta_sep * ax)
    ny = softplus(-ay, beta_sep)
    gy = sigmoid(-beta_sep * ay)
    mx = nx * gx
    my = ny * gy
    m_in = np.hypot(mx, my)

    dabsx = dx / np.sqrt(dx * dx + inv_beta_dir * inv_beta_dir)
    dabsy = dy / np.sqrt(dy * dy + inv_beta_dir * inv_beta_dir)
    dd_dpx = np.zeros_like(px)
    dd_dpy = np.zeros_like(py)
    mask = d_out > 0.0
    dd_dpx[mask] = px[mask] / d_out[mask]
    dd_dpy[mask] = py[mask] / d_out[mask]
    dpx_dax = sigmoid(beta_sep * ax)
    dpy_day = sigmoid(beta_sep * ay)

    dd_ddx = dd_dpx * dpx_dax * dabsx
    dd_ddy = dd_dpy * dpy_day * dabsy
    dd_drc = dd_ddx * ux + dd_ddy * uy
    f_out_mag = invdist_force_mag(d_eff, k_out, pwr) * dd_drc
    fx_out = f_out_mag * ux
    fy_out = f_out_mag * uy

    dm_dmx = np.zeros_like(mx)
    dm_dmy = np.zeros_like(my)
    mask_in = m_in > 0.0
    dm_dmx[mask_in] = mx[mask_in] / m_in[mask_in]
    dm_dmy[mask_in] = my[mask_in] / m_in[mask_in]

    dnx_dax = -gx
    dgx_dax = gx * (1.0 - gx) * (-beta_sep)
    dmx_dax = dnx_dax * gx + nx * dgx_dax
    dax_ddx = dabsx

    dny_day = -gy
    dgy_day = gy * (1.0 - gy) * (-beta_sep)
    dmy_day = dny_day * gy + ny * dgy_day
    day_ddy = dabsy

    dm_ddx = dm_dmx * dmx_dax * dax_ddx
    dm_ddy = dm_dmy * dmy_day * day_ddy
    dm_drc = dm_ddx * ux + dm_ddy * uy
    f_in_mag = -k_in * m_in * dm_drc
    fx_in = f_in_mag * ux
    fy_in = f_in_mag * uy

    fx = fx_out + fx_in
    fy = fy_out + fy_in

    return np.stack([fx, fy], axis=1)


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample label-label rectangle field at ``xy`` points."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    labels_xy = np.asarray(scene.get("labels_xy", []), float)
    WH = (
        np.asarray(scene.get("WH"), float)
        if scene.get("WH") is not None
        else np.zeros((labels_xy.shape[0], 2), float)
    )
    if labels_xy.size == 0:
        return np.zeros_like(xy, float)

    F = np.zeros_like(xy, float)
    for p, wh in zip(labels_xy, WH):
        if not (np.isfinite(p).all() and np.isfinite(wh).all()):
            continue
        F += _pairwise_force_rect(np.asarray(p, float), np.asarray(wh, float), xy, params)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("ll.rect")(probe)

