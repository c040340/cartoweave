# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps
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
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
)


@register("ll.disk")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    N = int(P.shape[0])
    eps = get_eps(cfg)
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "ll.disk"}

    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    llk = [get_ll_kernel(l) for l in labels]

    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    disk_mask = np.array([(k or "") == "disk" for k in llk], dtype=bool)
    mask = base_mask & disk_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "ll.disk")

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

    F = ensure_vec2(F, N)
    return float(E), F, {"term": "ll.disk", "ll.disk": src}
