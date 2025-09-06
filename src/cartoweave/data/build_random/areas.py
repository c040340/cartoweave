from __future__ import annotations

import numpy as np

from ..primitives.polygons import generate_polygon_by_area
from ..sampling.helpers import frame_metrics

__all__ = ["generate_areas"]


def generate_areas(params, rng: np.random.Generator):
    frame_size = (params["frame"]["width"], params["frame"]["height"])
    diag, area_total = frame_metrics(frame_size)
    cfg = params["random"]["area_gen"]
    n_areas = int(params["counts"]["areas"])
    if n_areas <= 0:
        return []
    area_inset = cfg["inset_margin_scale"] * diag
    edge_spacing = cfg["min_edge_spacing_scale"] * diag
    n_v_min = cfg.get("n_vertices_min", 6)
    n_v_max = cfg.get("n_vertices_max", 14)
    polys = []
    for _ in range(n_areas):
        s_target = 0.02 * area_total
        polys.append(
            generate_polygon_by_area(
                rng,
                frame_size,
                s_target,
                area_inset,
                edge_spacing,
                (n_v_min, n_v_max),
            )
        )
    return polys
