"""Spring force term ('anchor.spring'): pulls P toward resolved anchor A. Physics lives here; geometry resolution lives in resolver."""
from __future__ import annotations

import numpy as np

from cartoweave.compute.geom_anchor_resolver import anchor_position
from cartoweave.utils.compute_common import ensure_vec2, get_eps, weight_of
from cartoweave.utils.kernels import EPS_NORM
from cartoweave.utils.logging import logger

from . import register


def _val(lab, key, default=None):
    """通用字段读取：兼容 dict 和 LabelState。
       - 支持 'kind' / 'mode' / 其它 meta 字段（mode 会从 meta 提升）
    """
    if isinstance(lab, dict):
        if key == "mode":
            return lab.get("mode") or (lab.get("meta") or {}).get("mode", default)
        return lab.get(key, default)
    # LabelState
    if key == "mode":
        m = getattr(lab, "meta", None)
        return (m or {}).get("mode", default)
    return getattr(lab, key, default)


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, cfg: dict, phase: str):  # noqa: N803
    n = P.shape[0] if P is not None else 0
    eps = get_eps(cfg)
    w = weight_of("anchor.spring", cfg, 0.0)
    if phase != "anchor" or w <= 0.0 or P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    k = float(cfg.get("anchor.k.spring", 10.0))
    r0_points = float(cfg.get("anchor.r0.points", 0.0))
    eps_n = float(cfg.get("eps.norm", EPS_NORM))

    labels_all = scene.get("labels", [])
    active_ids = scene.get("_active_ids", list(range(n)))
    if len(active_ids) != n:
        raise ValueError(f"_active_ids misaligned: {len(active_ids)} vs P {P.shape}")
    labels = [labels_all[i] for i in active_ids]
    anchors = np.asarray([anchor_position(lbl, scene, P) for lbl in labels], dtype=float)
    modes = [_val(lab, "mode") for lab in labels]
    mask = np.array([m != "circle" for m in modes], dtype=bool)
    idxs = np.nonzero(mask)[0]
    skip_circle = int(np.count_nonzero(~mask))

    force = np.zeros_like(P)
    energy = 0.0
    ext_dir = scene.get("_ext_dir")
    if ext_dir is None:
        ext_dir = np.zeros((n, 2))
    else:
        ext_dir = np.asarray(ext_dir, float)
        if ext_dir.shape[0] != n:
            raise ValueError(f"ext_dir misaligned: {ext_dir.shape} vs P {P.shape}")

    for i in idxs:
        px, py = P[i]
        ax, ay = anchors[i]
        dx = px - ax
        dy = py - ay
        r = (dx * dx + dy * dy) ** 0.5
        r0 = r0_points
        if r > eps:
            u = np.array([dx, dy]) / (r + eps_n)
            coeff = -k * (r - r0)
            fx, fy = coeff * u
        else:
            u_ext = ext_dir[i]
            nrm = float(np.hypot(u_ext[0], u_ext[1]))
            if nrm > 0:
                u = u_ext / (nrm + eps_n)
            else:
                u = np.array([0.0, 0.0])
            coeff = -k * (0.0 - r0)
            fx, fy = coeff * u
        force[i, 0] += fx
        force[i, 1] += fy
        energy += 0.5 * k * (r - r0) ** 2

    logger.debug("term_anchor: skip_circle=%d", skip_circle)
    force = ensure_vec2(force, n)
    return float(energy * w), force * w, {"term": "anchor.spring", "n": int(n)}
