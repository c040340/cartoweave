"""Spring force pulling labels toward resolved anchors."""
from __future__ import annotations

import numpy as np

from cartoweave.compute.geom_anchor_resolver import anchor_position

from . import register
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

    N = int(P.shape[0])
    labels = read_labels_aligned(scene, P)
    modes = [get_mode(l) for l in labels]
    base_mask = np.array([(m or "").lower() != "circle" for m in modes], dtype=bool)
    mask = base_mask

    k = float(cfg.get("anchor.k", 0.0))
    if k <= 0.0:
        return 0.0, ensure_vec2(np.zeros_like(P), N), {"term": "anchor.spring", "disabled": True}

    anchors = np.asarray(scene.get("anchors"), float)
    if anchors.shape != (N, 2):
        anchors = np.asarray([anchor_position(l, scene, P) for l in labels], float)

    F = np.zeros_like(P, float)
    E = 0.0
    for i in range(N):
        if not mask[i]:
            continue
        dx = P[i, 0] - anchors[i, 0]
        dy = P[i, 1] - anchors[i, 1]
        E += 0.5 * k * (dx * dx + dy * dy)
        F[i, 0] -= k * dx
        F[i, 1] -= k * dy

    return float(E), ensure_vec2(F, N), {"term": "anchor.spring"}

