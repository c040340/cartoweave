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
       - 支持 'll_kernel' / 'mode' / 其它 meta 字段（mode 会从 meta 提升）
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
        if isinstance(a, dict):
            return {
                "kind": a["kind"] if "kind" in a else None,
                "index": a["index"] if "index" in a else None,
                "t": a["t"] if "t" in a else None,
            }
        return {
            "kind": getattr(a, "kind", None),
            "index": getattr(a, "index", None),
            "t": getattr(a, "t", None),
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

    import logging
    log = logging.getLogger(__name__)

    N = P.shape[0]
    labels_all = scene.get("labels", []) or []

    # 1) 与 P 完全对齐：优先 _active_ids_solver，其次 _active_ids；不匹配就回退为 0..N-1
    all_active = scene.get("_active_ids_solver") or scene.get("_active_ids")
    if not all_active or len(all_active) != N:
        log.warning(
            "ll.rect: active ids len=%s != P rows=%d; fallback to identity [0..N-1]",
            None if all_active is None else len(all_active), N
        )
        all_active = list(range(N))

    # 2) 与 P 同序取 labels（避免错位）；越界给空壳
    def _get_label(i):
        if 0 <= i < len(labels_all):
            return labels_all[i]
        class _Dummy:
            kind = None
            WH = (0.0, 0.0)
            def __getattr__(self, _): return None
        return _Dummy()

    labels = [_get_label(i) for i in all_active]

    # 3) WH 由 labels 同序组装，并稳健规范化到 (N,2)
    WH = np.asarray([getattr(lab, "WH", (0.0, 0.0)) for lab in labels], dtype=float)
    if WH.ndim != 2 or WH.shape != (N, 2):
        if WH.ndim == 0:
            WH = np.full((N, 2), float(WH))
        elif WH.ndim == 1 and WH.shape[0] == 2:
            WH = np.broadcast_to(WH.reshape(1, 2), (N, 2)).astype(float, copy=False)
        elif WH.ndim == 1 and WH.shape[0] == N:
            WH = np.stack([WH, WH], axis=1).astype(float, copy=False)
        else:
            log.warning("ll.rect: cannot normalize WH shape %s, fallback zeros (N,2)", WH.shape)
            WH = np.zeros((N, 2), dtype=float)

    # 4) 行内过滤：排除 circle，并根据 ll_kernel 选择参与者
    llk   = [getattr(lab, "ll_kernel", None) for lab in labels]
    modes = [_val(lab, "mode") for lab in labels]

    def _canon_llk(x):
        x = (x or "").lower()
        if x in ("rectangle",):
            return "rect"
        if x in ("ellipse", "elliptic", "circle"):
            return "disk"
        return x

    llk = [_canon_llk(x) for x in llk]

    # participate in ll.rect iff ll_kernel == "rect" AND real mode != "circle"
    # DO NOT exclude "area" here; area labels can still be rect-kernel participants if configured so.
    mask = np.array([(k == "rect") and ((m or "").lower() != "circle")
                     for k, m in zip(llk, modes)], dtype=bool)
    idxs  = np.nonzero(mask)[0]

    # ===== 你的原始主循环逻辑从这里继续（不变） =====
    F = np.zeros_like(P)
    E = 0.0

    k_out = float(cfg.get("ll.k.repulse", 0.0))
    k_in  = float(cfg.get("ll.k.inside", 0.0))
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
            b  = idxs[bi]
            xb, yb = float(P[b, 0]), float(P[b, 1])
            wb, hb = float(WH[b, 0]), float(WH[b, 1])

            dx, dy = xa - xb, ya - yb
            adx, ady = softabs(dx, eps_a), softabs(dy, eps_a)
            sx = adx - 0.5 * (wa + wb)
            sy = ady - 0.5 * (ha + hb)

            spx = softplus(sx, beta_sep)
            spy = softplus(sy, beta_sep)
            g   = (spx * spx + spy * spy + g_eps * g_eps) ** 0.5

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
                F[a, 0] += fx; F[a, 1] += fy
                F[b, 0] -= fx; F[b, 1] -= fy

            if k_in > 0.0:
                vin = softplus(-sx, beta_in) + softplus(-sy, beta_in)
                E += 0.5 * k_in * (vin * vin)
                c1 = k_in * vin * (-sigmoid(-beta_in * sx)) * (dx / (adx + eps))
                c2 = k_in * vin * (-sigmoid(-beta_in * sy)) * (dy / (ady + eps))
                fx_in = -c1; fy_in = -c2
                F[a, 0] += fx_in; F[a, 1] += fy_in
                F[b, 0] -= fx_in; F[b, 1] -= fy_in

    M = len(idxs)
    F = ensure_vec2(F, L)
    return float(E), F, {"term": "ll.rect", "pairs": int(M * (M - 1) // 2)}

