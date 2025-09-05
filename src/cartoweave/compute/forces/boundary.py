# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register
from cartoweave.utils.compute_common import get_eps, weight_of, ensure_vec2
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
from cartoweave.utils.geometry import project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger


@register("boundary.wall")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):
    L = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("boundary.wall", cfg, 0.0)
    if phase != "pre_anchor" or w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "boundary.wall"}

    W, H = scene.get("frame_size", (1920.0, 1080.0))
    W = float(W)
    H = float(H)

    N = P.shape[0]
    WH = np.asarray(scene.get("WH"), float)
    assert WH.shape[0] == N, f"WH misaligned: {WH.shape} vs P {P.shape}"
    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(N)))
    assert len(active_ids) == N, f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}"
    labels = [labels_all[i] if i < len(labels_all) else {} for i in active_ids]
    modes = [lab.get("mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]
    skip_circle = int(np.count_nonzero(~mask))

    k_wall = float(cfg.get("boundary.k.wall", 240.0))
    power = float(cfg.get("boundary.wall_power", 3.0))
    pad = float(cfg.get("boundary.pad", 0.0))
    beta_d = float(cfg.get("beta.softplus.dist", 3.0))
    eps_div = float(cfg.get("boundary.wall_eps", 0.3))
    k_in = float(cfg.get("boundary.k.in", 0.0))
    y_down = bool(cfg.get("boundary.y_down", True))
    topk = int(cfg.get("source.topk", 0))

    v0 = math.log(2.0) / max(beta_d, 1e-8)
    e0 = v0 + eps_div
    if k_in <= 0.0:
        k_in = k_wall / ((e0 ** power) * max(v0, 1e-8))

    F = np.zeros_like(P, float)
    E = 0.0
    src = [[] for _ in range(N)]

    def piece(s: float, ex: float, ey: float):
        c = softplus(s, beta_d) + eps_div
        v = softplus(-s, beta_d)
        sc = sigmoid(beta_d * s)
        sv = sigmoid(-beta_d * s)
        E_out = invdist_energy(c, k_wall, power)
        F_outm = invdist_force_mag(c, k_wall, power) * sc
        E_in = 0.5 * k_in * (v * v)
        F_inm = k_in * v * sv
        fm = (F_outm + F_inm)
        return (E_out + E_in), fm * ex, fm * ey, fm

    for i in idxs:
        w_i, h_i = float(WH[i, 0]), float(WH[i, 1])
        if w_i <= 0.0 and h_i <= 0.0:
            continue
        cx, cy = float(P[i, 0]), float(P[i, 1])
        hx, hy = 0.5 * w_i, 0.5 * h_i
        sL = (cx - hx) - pad
        sR = (W - pad) - (cx + hx)
        sT = (cy - hy) - pad
        sB = (H - pad) - (cy + hy)
        EL, fxL, fyL, gL = piece(sL, +1.0, 0.0)
        ER, fxR, fyR, gR = piece(sR, -1.0, 0.0)
        ET, fxT, fyT, gT = piece(sT, 0.0, +1.0)
        EB, fxB, fyB, gB = piece(sB, 0.0, -1.0)
        Fx = fxL + fxR + fxT + fxB
        Fy = fyL + fyR + fyT + fyB
        if not y_down:
            Fy = -Fy
        F[i, 0] += Fx
        F[i, 1] += Fy
        E += (EL + ER + ET + EB)
        src[i].append((0, float(fxL), float(fyL), float(gL)))
        src[i].append((1, float(fxR), float(fyR), float(gR)))
        src[i].append((2, float(fxT), float(fyT), float(gT)))
        src[i].append((3, float(fxB), float(fyB), float(gB)))

    if topk and topk > 0:
        pass
    logger.debug("term_boundary: skip_circle=%d", skip_circle)
    F = ensure_vec2(F, L)
    return float(E * w), F * w, {"term": "boundary.wall", "boundary": src}
