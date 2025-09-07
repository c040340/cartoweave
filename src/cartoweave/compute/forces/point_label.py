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


@register("pl.rect")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}
    pts = scene.get("points")
    if pts is None or len(pts) == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "pl.rect"}

    N = P.shape[0]
    WH = np.asarray(
        [getattr(lbl, "WH", (0.0, 0.0)) for lbl in scene.get("labels", [])],
        dtype=float
    )
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    pts = np.asarray(pts, float).reshape(-1, 2)
    N, M = P.shape[0], pts.shape[0]

    # --- 与 P 完全对齐：尽量使用活跃索引，否则回退为 0..N-1 ---
    labels_all = scene.get("labels", [])
    all_active = (scene.get("_active_ids_solver")
                  or scene.get("_active_ids"))

    N = P.shape[0]
    if not all_active or len(all_active) != N:
        # 兜底：身份映射，保证与 P 的行号一致
        import logging
        logging.getLogger(__name__).warning(
            "pl.rect: active ids len=%s != P rows=%d; fallback to identity [0..N-1]",
            None if all_active is None else len(all_active), N
        )
        all_active = list(range(N))

    # 与 P 同序取 labels（全体活跃，不做 kind 过滤，只在行内跳过非 point/circle）
    # 注意：labels_all 可能比 N 少；这种极端情况用空占位补齐
    def _get_label(i):
        if 0 <= i < len(labels_all):
            return labels_all[i]
        # 占位空壳（保持字段访问不炸）
        class _Dummy:
            kind = None
            WH = (0.0, 0.0)
            def __getattr__(self, _): return None
        return _Dummy()

    labels = [_get_label(i) for i in all_active]

    # WH 与 P 对齐（N,2）
    WH = np.asarray(
        [getattr(lab, "WH", (0.0, 0.0)) for lab in labels],
        dtype=float
    )
    if WH.ndim != 2 or WH.shape != (N, 2):
        # 进一步兜底：把任何异常形状都正规化为 (N,2)
        from numpy import full
        if WH.ndim == 0:
            WH = full((N, 2), float(WH))
        elif WH.ndim == 1 and WH.shape[0] == 2:
            WH = np.broadcast_to(WH.reshape(1, 2), (N, 2)).astype(float, copy=False)
        elif WH.ndim == 1 and WH.shape[0] == N:
            WH = np.stack([WH, WH], axis=1).astype(float, copy=False)
        else:
            # 实在对不齐就用 0 占位，保证不断
            import logging
            logging.getLogger(__name__).warning(
                "pl.rect: cannot normalize WH shape %s, fallback zeros (N,2)",
                WH.shape
            )
            WH = np.zeros((N, 2), dtype=float)

    # 只在“行内”跳过非 point 或 mode=="circle" 的行
    modes = [_val(lab, "mode") for lab in labels]
    kinds = [getattr(lab, "kind", None) for lab in labels]
    mask = np.array(
        [(k == "point") and (m != "circle") for k, m in zip(kinds, modes)],
        dtype=bool
    )
    idxs = np.nonzero(mask)[0]

    skip_circle = int(np.count_nonzero(~mask))
    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("pl.k.repulse", 0.0))
    k_in = float(cfg.get("pl.k.inside", 0.0))
    pwr = float(cfg.get("pl.edge_power", 2.0))
    eps_d = float(cfg.get("eps.dist", EPS_DIST))
    eps_a = float(cfg.get("eps.abs", EPS_ABS))

    beta_sep = float(cfg.get("pl.beta.sep", 6.0))
    beta_in = float(cfg.get("pl.beta.in", 6.0))
    g_eps = float(cfg.get("pl.g_eps", 1e-6))

    for i in idxs:
        lab = labels[i]
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        cx, cy = float(P[i, 0]), float(P[i, 1])
        for j in range(M):
            x, y = float(pts[j, 0]), float(pts[j, 1])
            dx, dy = cx - x, cy - y
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * w_i
            sy = ady - 0.5 * h_i

            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5

            if k_out > 0.0:
                E += invdist_energy(g + eps_d, k_out, pwr)
                dEdg = -invdist_force_mag(g + eps_d, k_out, pwr)
                if g > 0.0:
                    d_g_ddx = (spx / g) * sigmoid(beta_sep * sx) * (dx / (adx + eps))
                    d_g_ddy = (spy / g) * sigmoid(beta_sep * sy) * (dy / (ady + eps))
                    fx = -dEdg * d_g_ddx
                    fy = -dEdg * d_g_ddy
                else:
                    fx = fy = 0.0
                F[i, 0] += fx
                F[i, 1] += fy

            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                fx_in = -k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                fy_in = -k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                F[i, 0] += fx_in
                F[i, 1] += fy_in

    logger.debug("term_point_label: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E), F, {"term": "pl.rect", "pl": int(N * M)}
