# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from . import register, register_probe, term_cfg, eps_params
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
    return {"kind": getattr(a, "kind", None), "index": getattr(a, "index", None), "t": getattr(a, "t", None)}


def _focus_force_at(xy: np.ndarray, center: np.ndarray, sigma_xy: np.ndarray, delta: float, k: float) -> np.ndarray:
    """Closed-form focus attraction force at ``xy`` points."""

    xy = np.asarray(xy, dtype=float)
    rx = (xy[:, 0] - center[0]) / sigma_xy[0]
    ry = (xy[:, 1] - center[1]) / sigma_xy[1]
    Q = rx * rx + ry * ry
    root = np.sqrt(1.0 + Q / (delta * delta))
    denom = np.maximum(root, 1e-12)
    Fx = -k * rx / (sigma_xy[0] * denom)
    Fy = -k * ry / (sigma_xy[1] * denom)
    return np.stack([Fx, Fy], axis=1)


def _huges_force_at(
    xy: np.ndarray,
    center: np.ndarray,
    sigma_xy: np.ndarray,
    band: float,
    k: float,
    eps: float,
) -> np.ndarray:
    """
    椭圆壳力 F：壳内推外，壳外拉内，壳上为 0；r=1 附近 Huber 平滑。
    返回与 xy 同形状的 (M,2) 力向量。
    """
    sigx = max(float(sigma_xy[0]), eps)
    sigy = max(float(sigma_xy[1]), eps)
    bx = max(float(band), eps)

    rx = (xy[:, 0] - center[0]) / sigx
    ry = (xy[:, 1] - center[1]) / sigy
    r = np.sqrt(rx * rx + ry * ry)
    s = r - 1.0

    root = np.sqrt(1.0 + (s / bx) * (s / bx))
    dE_dr = k * s / np.maximum(root, eps)

    denom = np.maximum(r, eps)
    grad_r_x = rx / (sigx * denom)
    grad_r_y = ry / (sigy * denom)

    fx = -dE_dr * grad_r_x
    fy = -dE_dr * grad_r_y
    return np.stack([fx, fy], axis=1)


@register("focus.attract")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    N = int(P.shape[0])
    tc = term_cfg(cfg, "focus", "attract")
    epss = eps_params(cfg, tc, defaults={})
    eps = epss["eps_numeric"]

    k_attract = float(0.8 if tc.get("k_attract") is None else tc.get("k_attract"))
    intro_times = float(tc.get("intro_times", 4))
    k_huges = k_attract * intro_times
    if k_attract <= 0.0 and k_huges <= 0.0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "focus.attract"}

    center_rel = tc.get("center", None)
    if center_rel is None:
        center_rel = scene.get("focus_center", None)
    if center_rel is None:
        center_rel = (0.5, 0.5)
    W, H = scene.get("frame_size", (1920.0, 1080.0))
    W, H = float(W), float(H)

    labels = read_labels_aligned(scene, P)
    WH = normalize_WH_from_labels(labels, N, "focus.attract")
    modes = [get_mode(l) for l in labels]
    mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]

    F = np.zeros_like(P, float)
    E = 0.0
    info_focus = []
    info_ring = []

    if k_attract > 0.0:
        Cx = float(center_rel[0]) * W
        Cy = float(center_rel[1]) * H
        sigma = float(tc.get("sigma") or 1.0)
        wh = tc.get("wh")
        if wh is not None:
            wx, hy = float(wh[0]), float(wh[1])
            sigx = max(eps, sigma * wx * W)
            sigy = max(eps, sigma * hy * H)
        else:
            sigx = sigy = max(sigma, eps)
        delta = float(8.0 if tc.get("delta") is None else tc.get("delta"))
        only_free = bool(tc.get("only_free") if tc.get("only_free") is not None else False)

        center_xy = np.array([Cx, Cy], float)
        sig_xy = np.array([sigx, sigy], float)
        for i in idxs:
            lab = labels[i]
            if only_free:
                a = _anchor(lab)
                kind = a["kind"] if a and a["kind"] is not None else "none"
                if kind and kind != "none":
                    continue
            x, y = float(P[i, 0]), float(P[i, 1])
            Fi = _focus_force_at(np.array([[x, y]], float), center_xy, sig_xy, delta, k_attract)[0]
            rx = (x - Cx) / sigx
            ry = (y - Cy) / sigy
            Q = rx * rx + ry * ry
            root = (1.0 + Q / (delta * delta)) ** 0.5
            E_i = k_attract * (delta * delta) * (root - 1.0)
            fx, fy = float(Fi[0]), float(Fi[1])
            F[i, 0] += fx
            F[i, 1] += fy
            E += E_i
            info_focus.append((int(i), float(E_i), float(fx), float(fy)))

    if k_huges > 0.0:
        center_h = tc.get("center", center_rel)
        Cx_h = float(center_h[0]) * W
        Cy_h = float(center_h[1]) * H
        wh = tc.get("wh")
        if wh is not None:
            wx, hy = float(wh[0]), float(wh[1])
            a = max(eps, sigma * wx * W * 2)
            b = max(eps, sigma* hy * H * 2)
        else:
            a = b = max(sigma, eps)
        # a = float(100.0 if wh_cfg.get("x") is None else wh_cfg.get("x"))
        # b = float(100.0 if wh_cfg.get("y") is None else wh_cfg.get("y"))
        a = max(a, eps)
        b = max(b, eps)
        band = float(tc.get("band", tc.get("sigma", 0.1)))
        band = max(band, eps) * 4
        only_free_h = True #= bool(tc.get("only_free") if tc.get("only_free") is not None else False)

        for i in idxs:
            if only_free_h:
                ainfo = _anchor(labels[i])
                kind = ainfo["kind"] if ainfo and ainfo["kind"] is not None else "none"
                if kind and kind != "none":
                    continue
            xy = P[i:i + 1]
            fxy = _huges_force_at(
                xy=xy,
                center=np.array([Cx_h, Cy_h], dtype=float),
                sigma_xy=np.array([a, b], dtype=float),
                band=band,
                k=k_huges,
                eps=eps,
            )
            F[i, 0] += float(fxy[0, 0])
            F[i, 1] += float(fxy[0, 1])

            rx = (xy[0, 0] - Cx_h) / a
            ry = (xy[0, 1] - Cy_h) / b
            r = float(np.hypot(rx, ry))
            s = r - 1.0
            root = float(np.sqrt(1.0 + (s / band) * (s / band)))
            E_i = k_huges * (band * band) * (root - 1.0)
            E += E_i
            info_ring.append((int(i), float(E_i), float(fxy[0, 0]), float(fxy[0, 1]), r))

    F = ensure_vec2(F, N)
    meta = {"term": "focus.attract"}
    if info_focus:
        meta["focus_huber"] = info_focus
    if info_ring:
        meta["ring_huber"] = info_ring
    return float(E), F, meta




def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample focus force field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    frame_size = scene.get("frame_size", (1000.0, 1000.0))
    W, H = float(frame_size[0]), float(frame_size[1])
    eps = 1e-12

    center = params.get("center")
    if center is None:
        center = scene.get("focus_center")
    if center is None:
        center = (0.5, 0.5)
    Cx = float(center[0]) * W
    Cy = float(center[1]) * H

    k_attract = float_param(params, "k_attract", 0.8)
    intro_times = float_param(params, "intro_times", 4.0)
    k_huges = k_attract * intro_times

    sigma = float_param(params, "sigma", 1.0)
    wh = params.get("wh")
    if wh is not None:
        wx, hy = float(wh[0]), float(wh[1])
        sigx = max(eps, sigma * wx * W)
        sigy = max(eps, sigma * hy * H)
        a = max(eps, sigma * wx * W * 2.0)
        b = max(eps, sigma * hy * H * 2.0)
    else:
        sigx = sigy = max(sigma, eps)
        a = b = max(sigma, eps)

    delta = float_param(params, "delta", 8.0)

    F = _focus_force_at(
        xy,
        np.array([Cx, Cy], float),
        np.array([sigx, sigy], float),
        delta,
        k_attract,
    )

    if k_huges > 0.0:
        band = float_param(params, "band", float_param(params, "sigma", 0.1))
        band = max(band, eps) * 4.0
        F += _huges_force_at(
            xy=xy,
            center=np.array([Cx, Cy], float),
            sigma_xy=np.array([a, b], float),
            band=band,
            k=k_huges,
            eps=eps,
        )

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


register_probe("focus.attract")(probe)
