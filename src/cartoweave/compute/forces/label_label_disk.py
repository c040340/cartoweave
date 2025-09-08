# -*- coding: utf-8 -*-
from __future__ import annotations
import math
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
    softmin_weights,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.geometry import project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
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


@register("ll.disk")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.disk"}

    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"

    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids_solver", list(range(N)))
    assert len(active_ids) == N, f"_active_ids_solver misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    llk   = [getattr(lab, "ll_kernel", None) for lab in labels]
    modes = [_val(lab, "mode") for lab in labels]

    def _canon_llk(x):
        x = (x or "").lower()
        if x in ("ellipse", "elliptic"):  # normalize synonyms to "disk"
            return "disk"
        return x

    llk = [_canon_llk(x) for x in llk]

    # participate in ll.disk iff ll_kernel == "disk" AND real mode != "circle"
    mask = np.array([(k == "disk") and ((m or "").lower() != "circle")
                     for k, m in zip(llk, modes)], dtype=bool)
    idxs = np.nonzero(mask)[0]

    k_out = float(cfg.get("ll.k.repulse", 900.0))
    p = float(cfg.get("ll.edge_power", 2.0))
    eps_sep = float(cfg.get("ll.edge_eps", 0.5))
    beta = float(cfg.get("beta.softplus.sep", cfg.get("beta_softplus_sep", 6.0)))
    k_in = float(cfg.get("ll.k.inside", 0.0))
    radius_mode = str(cfg.get("ll.disk.mode", "max")).lower()

    def radius_from_wh(w: float, h: float) -> float:
        if radius_mode == "min":
            return 0.5 * min(w, h)
        if radius_mode == "avg":
            return 0.25 * (w + h)
        if radius_mode == "diag":
            return 0.5 * math.hypot(w, h)
        return 0.5 * max(w, h)

    v0 = math.log(2.0) / max(beta, 1e-8)
    e0 = v0 + eps_sep
    if k_in <= 0.0:
        k_in = k_out / ((e0 ** p) * max(v0, 1e-8))

    F = np.zeros_like(P)
    E = 0.0
    src = [[] for _ in range(N)]

    for ii in range(len(idxs)):
        i = idxs[ii]
        wi, hi = float(WH[i, 0]), float(WH[i, 1])
        xi, yi = float(P[i, 0]), float(P[i, 1])
        if wi <= 0.0 and hi <= 0.0:
            continue
        ri = radius_from_wh(wi, hi)
        for jj in range(ii + 1, len(idxs)):
            j = idxs[jj]
            wj, hj = float(WH[j, 0]), float(WH[j, 1])
            xj, yj = float(P[j, 0]), float(P[j, 1])
            rj = radius_from_wh(wj, hj)
            dx, dy = xi - xj, yi - yj
            rc = math.hypot(dx, dy) + eps
            s = rc - (ri + rj)
            c = softplus(s, beta) + eps_sep
            v = softplus(-s, beta)
            sc = sigmoid(beta * s)
            sv = sigmoid(-beta * s)
            E += invdist_energy(c, k_out, p) + 0.5 * k_in * (v * v)
            fmag = invdist_force_mag(c, k_out, p) * sc + (k_in * v * sv)
            ux, uy = dx / rc, dy / rc
            fx, fy = fmag * ux, fmag * uy
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy
            src[i].append((int(j), float(fx), float(fy), float(abs(fmag))))
            src[j].append((int(i), float(-fx), float(-fy), float(abs(fmag))))

    F = ensure_vec2(F, L)
    return float(E), F, {"term": "ll.disk", "ll.disk": src}
