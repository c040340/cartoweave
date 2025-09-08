# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg, kernel_params, eps_params
from cartoweave.utils.kernels import (
    softplus,
    sigmoid,
    invdist_energy,
    invdist_force_mag,
)
from cartoweave.utils.geometry import project_point_to_segment, poly_signed_area, rect_half_extent_along_dir
from cartoweave.utils.shape import as_nx2
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
    float_param,
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


@register("boundary.wall")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    tc = term_cfg(cfg, "boundary", "wall")
    epss = eps_params(cfg, tc, defaults={"abs": 1e-3})
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "boundary.wall"}

    W, H = scene.get("frame_size", (1920.0, 1080.0))
    W, H = float(W), float(H)

    labels = read_labels_aligned(scene, P)
    N = int(P.shape[0])
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask
    idxs = np.nonzero(mask)[0]

    WH = normalize_WH_from_labels(labels, N, "boundary.wall")

    # ===== 你原来的参数读取与主循环从这里继续 =====
    k_wall = float(tc.get("k_wall", 240.0))
    ker = kernel_params(tc, defaults={"model": "inv_pow", "exponent": 3.0, "soft_eps": 0.3})
    power = ker["kernel_exponent"]
    eps_div = ker["kernel_soft_eps"]
    beta_dict = tc.get("beta") or {}
    beta_d = float(3.0 if beta_dict.get("dist") is None else beta_dict.get("dist"))
    pad = float(0.0 if tc.get("pad") is None else tc.get("pad"))
    k_in = float(0.0 if tc.get("k_in") is None else tc.get("k_in"))
    y_down = bool(tc.get("y_down") if tc.get("y_down") is not None else True)
    topk   = int(cfg.get("source.topk", 0))

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
    F = ensure_vec2(F, N)
    return float(E), F, {"term": "boundary.wall", "boundary": src}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample boundary repulsion field at ``xy`` points."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    W, H = scene.get("frame_size", (1000.0, 1000.0))
    W, H = float(W), float(H)

    k_wall = float_param(params, "k_wall", 240.0)
    ker = kernel_params(params, defaults={"model": "inv_pow", "exponent": 3.0, "soft_eps": 0.3})
    power = ker["kernel_exponent"]
    eps_div = ker["kernel_soft_eps"]
    beta_d = float_param(params.get("beta") or {}, "dist", 3.0)
    pad = float_param(params, "pad", 0.0)
    k_in = float_param(params, "k_in", 0.0)
    y_down = bool(params.get("y_down", True))

    sL = xy[:, 0] - pad
    sR = (W - pad) - xy[:, 0]
    sT = xy[:, 1] - pad
    sB = (H - pad) - xy[:, 1]

    def piece(s: np.ndarray) -> np.ndarray:
        c = softplus(s, beta_d) + eps_div
        v = softplus(-s, beta_d)
        sc = sigmoid(beta_d * s)
        sv = sigmoid(-beta_d * s)
        F_outm = invdist_force_mag(c, k_wall, power) * sc
        F_inm = k_in * v * sv
        return F_outm + F_inm

    Fx = piece(sL) - piece(sR)
    Fy = piece(sT) - piece(sB)
    if not y_down:
        Fy = -Fy
    F = np.stack([Fx, Fy], axis=1)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("boundary.wall")(probe)
