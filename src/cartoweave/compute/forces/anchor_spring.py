"""Spring force pulling labels toward resolved anchors."""
from __future__ import annotations

import math
import numpy as np

from cartoweave.compute.geom_anchor_resolver import anchor_position

from . import register, register_probe, term_cfg, kernel_params
from ._common import (
    read_labels_aligned,
    get_mode,
    get_ll_kernel,
    normalize_WH_from_labels,
    ensure_vec2,
)


@register("anchor.spring")
def evaluate(scene: dict, P: np.ndarray, params: dict, cfg: dict):
    if P is None or P.size == 0:
        return 0.0, np.zeros_like(P), {"disabled": True, "term": "anchor.spring"}

    tc = term_cfg(cfg, "anchor", "spring")
    kernel_params(tc, defaults={"model": "poly", "exponent": 2.0, "soft_eps": 0.0})

    zero_dist = float(0.0 if tc.get("zero_dist") is None else tc.get("zero_dist"))
    zero_dist = max(0.0, zero_dist)

    N = int(P.shape[0])
    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask

    k_local = float(1.0 if tc.get("k_local") is None else tc.get("k_local"))
    if k_local <= 0.0:
        return 0.0, ensure_vec2(np.zeros_like(P), N), {"term": "anchor.spring", "disabled": True}

    anchors = np.asarray(scene.get("anchors"), float)
    if anchors.shape != (N, 2):
        anchors = np.asarray([anchor_position(l, scene, P) for l in labels], float)

    F = np.zeros_like(P, float)
    E = 0.0
    for i in range(N):
        if not mask[i]:
            continue
        dx = float(P[i, 0] - anchors[i, 0])
        dy = float(P[i, 1] - anchors[i, 1])
        dist = math.hypot(dx, dy)
        if dist <= zero_dist:
            continue
        d = dist - zero_dist
        E += 0.5 * k_local * (d * d)
        scale = -k_local * d / max(dist, 1e-12)
        F[i, 0] += scale * dx
        F[i, 1] += scale * dy

    return float(E), ensure_vec2(F, N), {"term": "anchor.spring"}


def probe(scene: dict, params: dict, xy: np.ndarray) -> np.ndarray:
    """Sample ``anchor.spring`` field at world coordinates ``xy``."""

    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise AssertionError("xy must be (M,2)")

    anchors = np.asarray(scene.get("anchors", []), float)
    if anchors.size == 0:
        return np.zeros_like(xy, float)

    k_local = float(1.0 if params.get("k_local") is None else params.get("k_local"))
    if k_local <= 0.0:
        return np.zeros_like(xy, float)

    zero_dist = float(0.0 if params.get("zero_dist") is None else params.get("zero_dist"))
    zero_dist = max(0.0, zero_dist)

    F = np.zeros_like(xy, float)
    for ax, ay in anchors:
        dx = xy[:, 0] - float(ax)
        dy = xy[:, 1] - float(ay)
        dist = np.hypot(dx, dy)
        d = np.maximum(dist - zero_dist, 0.0)
        scale = -k_local * d / np.maximum(dist, 1e-12)
        F[:, 0] += scale * dx
        F[:, 1] += scale * dy

    if not np.isfinite(F).all():
        raise ValueError("anchor.spring probe produced non-finite values")
    return F


register_probe("anchor.spring")(probe)

