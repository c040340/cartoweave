from __future__ import annotations

import math

import numpy as np

from cartoweave.utils.kernels import (
    invdist_energy,
    invdist_force_mag,
    sigmoid,
    softplus,
)

from . import kernel_params, register, register_probe, term_cfg
from ._common import (
    active_element_indices,
    ensure_vec2,
    float_param,
    get_mode,
    read_labels_aligned,
)


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
def evaluate(scene: dict, p: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "pl", "rect")
    if p is None or p.size == 0:
        return 0.0, np.zeros_like(p), {"disabled": True, "term": "pl.rect"}
    pts_all = scene.get("points") or []
    if len(pts_all) == 0:
        return 0.0, np.zeros_like(p), {"disabled": True, "term": "pl.rect"}

    labels = read_labels_aligned(scene, p)
    n = int(p.shape[0])
    modes = [get_mode(lb) for lb in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask

    active_ids = scene.get("_active_ids_solver")
    if active_ids is not None:
        active_mask = np.zeros(n, dtype=bool)
        active_mask[np.asarray(active_ids, dtype=int)] = True
        mask &= active_mask

    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return 0.0, np.zeros_like(p), {"disabled": True, "term": "pl.rect"}

    active_pts = active_element_indices([labels[i] for i in idxs], "point")
    pt_idxs = sorted(active_pts)
    pts = [pts_all[i] for i in pt_idxs if 0 <= i < len(pts_all)]
    if not pts:
        return 0.0, np.zeros_like(p), {"disabled": True, "term": "pl.rect"}
    pts = np.asarray(pts, float).reshape(-1, 2)
    m = pts.shape[0]

    force = np.zeros_like(p)
    energy = 0.0

    k_out = float(0.8 if tc.get("k_out") is None else tc.get("k_out"))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 2.0, "soft_eps": 1e-6})
    pwr = ker["kernel_exponent"]
    eps_d = ker["kernel_soft_eps"]
    beta_cfg = tc.get("beta") or {}
    beta_d = beta_cfg.get("dist")
    if beta_d is None:
        beta_d = beta_cfg.get("sep")
    beta_d = float(6.0 if beta_d is None else beta_d)
    eps_norm = float(1e-9 if (tc.get("eps", {}).get("norm") is None) else tc.get("eps", {}).get("norm"))

    for i in idxs:
        lab = labels[i]
        anch = _anchor(lab)
        cx, cy = float(p[i, 0]), float(p[i, 1])
        for pj, pt_idx in enumerate(pt_idxs):
            if anch and anch.get("kind") == "point" and int(anch.get("index", -1)) == int(pt_idx):
                continue
            px, py = float(pts[pj, 0]), float(pts[pj, 1])
            dx, dy = cx - px, cy - py
            r_raw = math.hypot(dx, dy)
            if r_raw > eps_norm:
                ux, uy = dx / r_raw, dy / r_raw
            else:
                ux = uy = 0.0
            d_eff = softplus(r_raw, beta_d) + eps_d
            sdr = sigmoid(beta_d * r_raw)
            fmag = invdist_force_mag(d_eff, k_out, pwr) * sdr
            force[i, 0] += fmag * ux
            force[i, 1] += fmag * uy
            energy += invdist_energy(d_eff, k_out, pwr)

    force = ensure_vec2(force, n)
    return float(energy), force, {"term": "pl.rect", "pl": int(n * m)}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``pl.rect`` field caused by static points."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    pts = scene.get("points")
    if pts is None or len(pts) == 0:
        return np.zeros_like(xy, float)

    k_out = float_param(params, "k_out", 0.8)
    beta_d = float_param(params, "beta", 6.0)
    eps_d = float_param(params, "eps_d", 1e-6)
    pts = np.asarray(pts, float).reshape(-1, 2)

    dx = xy[:, None, 0] - pts[None, :, 0]
    dy = xy[:, None, 1] - pts[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    d_eff = softplus(dist, beta_d) + eps_d
    sdr = sigmoid(beta_d * dist)
    invr = np.divide(1.0, dist, out=np.zeros_like(dist), where=dist > 0.0)
    mag = k_out / (d_eff ** 2) * sdr
    fx = (mag * dx * invr).sum(axis=1)
    fy = (mag * dy * invr).sum(axis=1)
    force = np.stack([fx, fy], axis=1)

    force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)
    return force


register_probe("pl.rect")(probe)
