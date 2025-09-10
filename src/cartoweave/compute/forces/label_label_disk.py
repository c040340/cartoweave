# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
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


def radius_from_wh(w: float, h: float, mode: str = "max") -> float:
    mode = mode.lower()
    if mode == "min":
        return 0.5 * min(w, h)
    if mode == "avg":
        return 0.25 * (w + h)
    if mode == "diag":
        return 0.5 * math.hypot(w, h)
    return 0.5 * max(w, h)


@register("ll.disk")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    N = int(P.shape[0])
    tc = term_cfg(cfg, "ll", "disk")
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.disk"}

    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]

    # If the rectangle kernel is disabled, treat all non-circle labels as disks
    # regardless of their ``ll_kernel`` attribute.  This mirrors the behaviour
    # where a single implementation is selected globally via YAML and avoids
    # missing forces when labels default to ``ll_kernel='rect'``.
    rect_cfg = term_cfg(cfg, "ll", "rect")
    rect_enabled = bool(rect_cfg.get("enable", False))
    if rect_enabled:
        llk = [get_ll_kernel(l) for l in labels]
        disk_mask = np.array([(k or "") == "disk" for k in llk], dtype=bool)
    else:
        disk_mask = np.ones(N, dtype=bool)

    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask & disk_mask

    active_ids = scene.get("_active_ids_solver")
    if active_ids is not None:
        active_mask = np.zeros(N, dtype=bool)
        active_mask[np.asarray(active_ids, dtype=int)] = True
        mask &= active_mask

    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "ll.disk")

    k_out = float(tc.get("k_ll_repulse", 900.0))
    pwr = float(tc.get("ll_edge_power", 2.0))
    eps_sep = float(tc.get("ll_edge_eps", 0.5))
    beta = float(tc.get("beta_softplus_sep", 6.0))
    radius_mode = str((tc.get("ll_disk_mode", "max")).lower())
    v0 = math.log(2.0) / max(beta, 1e-8)
    e0 = v0 + eps_sep
    k_in_auto = k_out / ((e0 ** pwr) * max(v0, 1e-8))
    k_in_val = tc.get("k_ll_inside")
    if k_in_val is None:
        k_in_val = k_in_auto
    k_in = float(k_in_val)
    eps_r = float(tc.get("ll_center_eps", 1e-3))

    F = np.zeros_like(P)
    E = 0.0
    src = [[] for _ in range(N)]

    for ii in range(len(idxs)):
        i = idxs[ii]
        wi, hi = float(WH[i, 0]), float(WH[i, 1])
        xi, yi = float(P[i, 0]), float(P[i, 1])
        if wi <= 0.0 and hi <= 0.0:
            continue
        ri = radius_from_wh(wi, hi, radius_mode)
        for jj in range(ii + 1, len(idxs)):
            j = idxs[jj]
            wj, hj = float(WH[j, 0]), float(WH[j, 1])
            xj, yj = float(P[j, 0]), float(P[j, 1])
            rj = radius_from_wh(wj, hj, radius_mode)
            dx, dy = xi - xj, yi - yj
            rc = math.sqrt(dx * dx + dy * dy + eps_r * eps_r)
            s = rc - (ri + rj)
            c = softplus(s, beta) + eps_sep
            v = softplus(-s, beta)
            sc = sigmoid(beta * s)
            sv = sigmoid(-beta * s)
            E += invdist_energy(c, k_out, pwr) + 0.5 * k_in * (v * v)
            fmag = invdist_force_mag(c, k_out, pwr) * sc + (k_in * v * sv)
            ux, uy = dx / rc, dy / rc
            fx, fy = fmag * ux, fmag * uy
            F[i, 0] += fx
            F[i, 1] += fy
            F[j, 0] -= fx
            F[j, 1] -= fy
            ftot = math.hypot(fx, fy)
            src[i].append((int(j), float(fx), float(fy), float(ftot), float(c)))
            src[j].append((int(i), float(-fx), float(-fy), float(ftot), float(c)))

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "ll.disk", "ll.disk": src}


def _pairwise_force_disk(src_xy: np.ndarray, src_r: float, xy: np.ndarray, params: dict) -> np.ndarray:
    """Force from a disk ``src_xy`` acting on sample points ``xy``."""

    xy = np.asarray(xy, float)
    dx = xy[:, 0] - src_xy[0]
    dy = xy[:, 1] - src_xy[1]
    rc = np.sqrt(dx * dx + dy * dy + float_param(params, "ll_center_eps", 1e-3) ** 2)
    s = rc - src_r
    beta = float_param(params, "beta_softplus_sep", 6.0)
    pwr = float_param(params, "ll_edge_power", 2.0)
    eps_sep = float_param(params, "ll_edge_eps", 0.5)
    k_out = float_param(params, "k_ll_repulse", 900.0)
    k_in = float_param(params, "k_ll_inside", float("nan"))
    if not np.isfinite(k_in) or k_in <= 0.0:
        v0 = math.log(2.0) / max(beta, 1e-8)
        e0 = v0 + eps_sep
        k_in = k_out / ((e0 ** pwr) * max(v0, 1e-8))
    c = softplus(s, beta) + eps_sep
    v = softplus(-s, beta)
    sc = sigmoid(beta * s)
    sv = sigmoid(-beta * s)
    fmag = invdist_force_mag(c, k_out, pwr) * sc + (k_in * v * sv)
    fx = fmag * dx / rc
    fy = fmag * dy / rc
    return np.stack([fx, fy], axis=1)


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``ll.disk`` field induced by existing labels."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    labels_xy = np.asarray(scene.get("labels_xy", []), float)
    WH = np.asarray(scene.get("WH"), float) if scene.get("WH") is not None else np.zeros((labels_xy.shape[0], 2), float)
    if labels_xy.size == 0:
        return np.zeros_like(xy, float)

    F = np.zeros_like(xy, float)
    radius_mode = str((params.get("mode") or "max")).lower()
    for p, wh in zip(labels_xy, WH):
        if not (np.isfinite(p).all() and np.isfinite(wh).all()):
            continue
        r = radius_from_wh(float(wh[0]), float(wh[1]), radius_mode)
        F += _pairwise_force_disk(np.asarray(p, float), r, xy, params)

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("ll.disk")(probe)
