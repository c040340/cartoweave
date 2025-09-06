from __future__ import annotations

import numpy as np

from ..primitives.lines import generate_polyline_by_length
from ..sampling.helpers import frame_metrics

__all__ = ["generate_lines"]


def generate_lines(params, rng: np.random.Generator):
    frame_size = (params["frame"]["width"], params["frame"]["height"])
    diag, _ = frame_metrics(frame_size)
    cfg = params["random"]["route_gen"]
    n_lines = int(params["counts"]["lines"])
    if n_lines <= 0:
        return []
    seg_len0 = cfg["segment_len_scale"] * diag
    min_spacing = cfg["min_vertex_spacing_scale"] * diag
    inset = cfg["inset_margin_scale"] * diag
    lines = []
    for _ in range(n_lines):
        target_len = seg_len0 * rng.uniform(4.0, 6.0)
        lines.append(
            generate_polyline_by_length(
                rng, frame_size, target_len, min_spacing, inset, seg_len0
            )
        )
    return lines
