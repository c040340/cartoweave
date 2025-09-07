# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, ensure_vec2
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


@register("ll.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.rect"}

    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"

    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids_solver", list(range(N)))
    assert len(active_ids) == N, f"_active_ids_solver misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [_val(lab, "mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("ll.k.repulse", 0.0))
    k_in = float(cfg.get("ll.k.inside", 0.0))
    pwr = float(cfg.get("ll.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_n = float(cfg.get("eps.norm", EPS_NORM))
    eps_a = float(cfg.get("eps.abs", EPS_ABS))

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

    M = len(idxs)
    F = ensure_vec2(F, L)
    return float(E), F, {"term": "ll.rect", "pairs": int(M * (M - 1) // 2)}
