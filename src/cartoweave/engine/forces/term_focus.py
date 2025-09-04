# src/cartoweave/engine/forces/term_focus.py
from __future__ import annotations
import math
import numpy as np
from . import register

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

from cartoweave.utils.geometry import (
    project_point_to_segment,
    poly_signed_area,
    rect_half_extent_along_dir,
)
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger

@register("focus.attract")
def term_focus_huber(scene, P: np.ndarray, cfg, phase="pre_anchor"):
    """
    椭圆 pseudo-Huber 井（各向异性 + C^1 平滑）：
      E = k * δ^2 * ( sqrt(1 + Q/δ^2 ) - 1 ),
      Q = ((x-cx)/σx)^2 + ((y-cy)/σy)^2
      ∂E/∂x = k * (r_x / (σx * sqrt(1 + Q/δ^2)))
      其中 r_x = (x - cx)/σx, r_y 同理；力 F = -∇E

    配置键（cfg）：
      - focus.k.attract : 井强度 k（>0 启用）
      - focus.center    : 中心 np.array([cx, cy])（若未给，则尝试 scene["focus_center"]，再退化为画幅中心）
      - focus.sigma.x   : σx（默认 100.0）
      - focus.sigma.y   : σy（默认 100.0）
      - focus.delta     : δ（默认 8.0）
      - focus.only_free : 仅作用于未绑定的 label（默认 False）
    """
    if phase != "pre_anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    k = float(cfg.get("focus.k.attract", 0.0))
    if k <= 0.0:
        return 0.0, np.zeros_like(P), {}

    # 读取中心
    center = cfg.get("focus.center", None)
    if center is None:
        center = scene.get("focus_center", None)
    if center is None:
        # 退化为画幅中心
        fw, fh = scene.get("frame_size", (0.0, 0.0))
        center = np.array([float(fw) * 0.5, float(fh) * 0.5], dtype=float)
    Cx, Cy = float(center[0]), float(center[1])

    sigx = float(cfg.get("focus.sigma.x", 100.0))
    sigy = float(cfg.get("focus.sigma.y", 100.0))
    delta = float(cfg.get("focus.delta", 8.0))
    only_free = bool(cfg.get("focus.only_free", False))

    if sigx <= 1e-12 or sigy <= 1e-12:
        # 极端防护：σ 不能为 0
        sigx = max(sigx, 1e-12)
        sigy = max(sigy, 1e-12)

    labels = scene.get("labels", [])
    N = P.shape[0]
    WH_raw = scene.get("WH")
    if WH_raw is None:
        WH_raw = np.zeros((N, 2))
    WH = as_nx2(WH_raw, N, "WH")

    F = np.zeros_like(P, float)
    E = 0.0
    info = []
    skip_hidden = 0
    skip_circle = 0

    for i in range(N):
        lab = labels[i] if i < len(labels) else {}
        w, h = float(WH[i, 0]), float(WH[i, 1])
        if lab.get("hidden"):
            assert w <= 0.0 and h <= 0.0
            skip_hidden += 1
            continue
        if lab.get("mode") == "circle":
            assert abs(w - h) < 1e-9
            skip_circle += 1
            continue
        if only_free:
            kind = lab.get("anchor_kind", "none")
            if kind and kind != "none":
                continue

        x, y = float(P[i, 0]), float(P[i, 1])
        rx = (x - Cx) / sigx
        ry = (y - Cy) / sigy
        Q = rx*rx + ry*ry

        root = (1.0 + Q / (delta*delta)) ** 0.5
        E_i = k * (delta*delta) * (root - 1.0)

        # ∂E/∂x, ∂E/∂y
        denom = max(root, 1e-12)
        dEdx = k * (rx / (sigx * denom))
        dEdy = k * (ry / (sigy * denom))

        fx = -dEdx
        fy = -dEdy

        F[i,0] += fx
        F[i,1] += fy
        E += E_i
        info.append((i, float(E_i), float(fx), float(fy)))

    logger.debug(
        "term_focus_huber: skip_hidden=%d skip_circle=%d", skip_hidden, skip_circle
    )
    return float(E), F, {"focus_huber": info}