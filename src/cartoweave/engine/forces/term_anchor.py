# src/cartoweave/engine/forces/term_anchor.py
from __future__ import annotations
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
from cartoweave.utils.shape import as_nx2
from cartoweave.utils.logging import logger

@register("anchor.spring")
def term_anchor(scene, P: np.ndarray, cfg, phase="anchor"):
    if phase != "anchor" or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {}

    # 需要每个 label 的锚点位置（A）：scene["anchors"] 形状 (N,2)
    A = scene.get("anchors")
    if A is None:
        # 没给就不施力
        return 0.0, np.zeros_like(P), {}

    k = float(cfg.get("anchor.k.spring", 10.0))
    # [patch-yx-2025-09-05] fallback rest_offset_px from physics.quick.point_spring
    q = cfg.get("physics", {}).get("quick", {})
    r0_points = float(cfg.get("anchor.r0.points", q.get("point_spring", {}).get("rest_offset_px", 0.0)))  # [patch-yx-2025-09-05]
    r0_lines = float(cfg.get("anchor.r0.lines", q.get("line_spring", {}).get("rest_offset_px", 0.0)))  # [patch-yx-2025-09-05]
    r0_areas = float(cfg.get("anchor.r0.areas", q.get("area_spring", {}).get("rest_offset_px", 0.0)))  # [patch-yx-2025-09-05]
    alpha = float(cfg.get("anchor.spring.alpha", 1.0))    # 弹簧形状参数（占位，可不用）

    eps_n = float(cfg.get("eps.norm", EPS_NORM))
    N = P.shape[0]
    A = np.asarray(A, float)
    assert A.shape[0] == N, f"anchors misaligned: {A.shape} vs P {P.shape}"
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
    F = np.zeros_like(P)
    E = 0.0

    # 兜底方向：来自 pre_anchor 阶段的外力合力
    ext_dir = scene.get("_ext_dir")
    if ext_dir is None:
        ext_dir = np.zeros((N, 2))
    else:
        ext_dir = np.asarray(ext_dir, float)
        assert ext_dir.shape[0] == N, f"ext_dir misaligned: {ext_dir.shape} vs P {P.shape}"

    for i in idxs:
        lab = labels[i]
        w, h = float(WH[i, 0]), float(WH[i, 1])

        px, py = P[i]
        ax, ay = A[i]
        dx = px - ax
        dy = py - ay
        r = (dx*dx + dy*dy) ** 0.5

        r0 = r0_points  # 你后续可根据类型切换 r0

        if r > 1e-12:
            # 简单 Hooke：E=0.5*k*(r-r0)^2，F = -k*(r-r0)*u
            u = np.array([dx, dy]) / (r + eps_n)
            coeff = -k * (r - r0)
            fx, fy = coeff * u
        else:
            # r≈0 用外力方向兜底（与旧工程一致的稳定器）
            u_ext = ext_dir[i]
            nrm = float(np.hypot(u_ext[0], u_ext[1]))
            if nrm > 0:
                u = u_ext / (nrm + eps_n)
            else:
                u = np.array([0.0, 0.0])
            coeff = -k * (0.0 - r0)
            fx, fy = coeff * u

        F[i,0] += fx
        F[i,1] += fy
        # 能量
        E += 0.5 * k * (r - r0) ** 2

    logger.debug("term_anchor: skip_circle=%d", skip_circle)
    return float(E), F, {"n": int(N)}
