# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, weight_of, ensure_vec2
from cartoweave.utils.geometry import (
    project_point_to_segment,
    rect_half_extent_along_dir,
    polylines_to_segments,
)
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


def _val(lab, key, default=None):
    """通用字段读取：兼容 dict 和 LabelState。
       - 支持 'kind' / 'mode' / 其它 meta 字段（mode 会从 meta 提升）
    """
    if isinstance(lab, dict):
        if key == "mode":
            return lab.get("mode") or (lab.get("meta") or {}).get("mode", default)
        return lab.get(key, default)
    if key == "mode":
        m = getattr(lab, "meta", None)
        return (m or {}).get("mode", default)
    return getattr(lab, key, default)


def _WH(lab):
    """统一尺寸读取：返回 np.array([w, h])。"""
    v = lab["WH"] if isinstance(lab, dict) else getattr(lab, "WH", None)
    return np.asarray(v, dtype=float)


def _anchor(lab):
    """统一锚读取：返回 dict {'kind': ..., 'index': ..., 't': ...} 或 None。"""
    if isinstance(lab, dict):
        a = lab.get("anchor")
        if a is None:
            return None
        return {
            "kind": a.get("kind") if isinstance(a, dict) else None,
            "index": a.get("index") if isinstance(a, dict) else None,
            "t": a.get("t") if isinstance(a, dict) else None,
        }
    a = getattr(lab, "anchor", None)
    if a is None:
        return None
    return {
        "kind": getattr(a, "kind", None),
        "index": getattr(a, "index", None),
        "t": getattr(a, "t", None),
    }


@register("ln.rect")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("ln.rect", cfg, 0.0)
    if phase != "pre_anchor" or w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}
    segs_raw = scene.get("lines")
    if segs_raw is None or len(segs_raw) == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ln.rect"}
    segs_arr = polylines_to_segments(segs_raw)
    segs = segs_arr.reshape((segs_arr.shape[0], 4))
    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [_val(lab, "mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]
    skip_circle = int(np.count_nonzero(~mask))
    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("ln.k.repulse", 0.0))
    k_in = float(cfg.get("ln.k.inside", 0.0))
    pwr = float(cfg.get("ln.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_a = float(cfg.get("eps.abs", EPS_ABS))
    beta_sep = float(cfg.get("ln.beta.sep", 6.0))
    beta_in = float(cfg.get("ln.beta.in", 6.0))
    g_eps = float(cfg.get("ln.g_eps", 1e-6))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for s in segs:
            ax, ay, bx, by = map(float, s)
            qx, qy, t, tx, ty = project_point_to_segment(cx, cy, ax, ay, bx, by)
            nx, ny = -ty, tx
            dx, dy = (cx - qx), (cy - qy)
            r_n = rect_half_extent_along_dir(w_i, h_i, nx, ny, eps_abs=eps_a)
            s_n = softabs(nx * dx + ny * dy, eps_a) - r_n
            s_t = softabs(tx * dx + ty * dy, eps_a)
            spn = softplus(s_n, beta_sep)
            spt = softplus(s_t, beta_sep)
            g = (spn * spn + spt * spt + g_eps * g_eps) ** 0.5
            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_dC_n = (spn / g) * sigmoid(beta_sep * s_n)
                    d_g_dC_t = (spt / g) * sigmoid(beta_sep * s_t)
                    coeff_n = (nx * dx + ny * dy) / (softabs(nx * dx + ny * dy, eps_a) + eps)
                    coeff_t = (tx * dx + ty * dy) / (softabs(tx * dx + ty * dy, eps_a) + eps)
                    dC_x = d_g_dC_n * coeff_n * nx + d_g_dC_t * coeff_t * tx
                    dC_y = d_g_dC_n * coeff_n * ny + d_g_dC_t * coeff_t * ty
                    fx = -dEdg * dC_x
                    fy = -dEdg * dC_y
                else:
                    fx = fy = 0.0
                F[i, 0] += fx
                F[i, 1] += fy

            if k_in > 0.0:
                vin = softplus(-s_n, beta_in)
                E += 0.5 * k_in * (vin * vin)
                coeff_n = (nx * dx + ny * dy) / (softabs(nx * dx + ny * dy, eps_a) + eps)
                dE_dC_n = k_in * vin * (-sigmoid(-beta_in * s_n)) * coeff_n
                fx_in = -dE_dC_n * nx
                fy_in = -dE_dC_n * ny
                F[i, 0] += fx_in
                F[i, 1] += fy_in

    logger.debug("term_line_label: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"term": "ln.rect", "ln": int(segs.shape[0])}
