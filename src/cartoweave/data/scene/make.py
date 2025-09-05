"""High level scene generation orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ..sampling.helpers import frame_metrics, inset_rect
from ..primitives.lines import generate_polyline_by_length
from ..primitives.polygons import generate_polygon_by_area
from ..primitives.textspec import label_specs_for_len
from ..config import RouteGenCfg, AreaGenCfg
from .cache import save_scene, load_scene, ensure_script


def generate_scene(
    rng: np.random.Generator,
    frame_size: tuple[int, int],
    route_cfg: RouteGenCfg,
    area_cfg: AreaGenCfg,
    n_points_range: tuple[int, int],
    n_lines_range: tuple[int, int],
    n_polys_range: tuple[int, int],
    n_labels: int,
    n_steps: int,
    cache_path: str | None = None,
) -> Dict:
    """Generate a full scene payload.

    Points, lines and polygons are sampled inside an inset rectangle.  Label
    positions are drawn uniformly and accompanied with rough width estimates.
    If ``cache_path`` is supplied and exists the cached payload is returned.
    """
    if cache_path and Path(cache_path).exists():
        data = load_scene(cache_path)
        return ensure_script(data)

    D, A = frame_metrics(frame_size)
    n_points = int(rng.integers(n_points_range[0], n_points_range[1] + 1))
    n_lines = int(rng.integers(n_lines_range[0], n_lines_range[1] + 1))
    n_polys = int(rng.integers(n_polys_range[0], n_polys_range[1] + 1))

    # Points
    inset = route_cfg.inset_margin_scale * D
    rect = inset_rect(frame_size, inset)
    pts = rng.uniform([rect[0], rect[1]], [rect[2], rect[3]], size=(n_points, 2))

    # Lines
    lines = []
    L0 = route_cfg.segment_len_scale * D
    min_spacing = route_cfg.min_vertex_spacing_scale * D
    for _ in range(n_lines):
        L_target = L0 * rng.uniform(4.0, 6.0)
        lines.append(
            generate_polyline_by_length(
                rng,
                frame_size,
                L_target,
                min_spacing,
                inset,
                L0,
            )
        )

    # Polygons
    area_inset = area_cfg.inset_margin_scale * D
    edge_spacing = area_cfg.min_edge_spacing_scale * D
    polys = []
    for _ in range(n_polys):
        S_target = 0.02 * A
        polys.append(
            generate_polygon_by_area(
                rng,
                frame_size,
                S_target,
                area_inset,
                edge_spacing,
                (area_cfg.n_vertices_min, area_cfg.n_vertices_max),
            )
        )

    # Labels
    labels_init = rng.uniform([rect[0], rect[1]], [rect[2], rect[3]], size=(n_labels, 2))
    WH = np.array(
        [[label_specs_for_len(10).single_px, 24.0] for _ in range(n_labels)],
        dtype=float,
    )

    payload: Dict = {
        "points": pts,
        "lines": lines,
        "areas": polys,
        "labels_init": labels_init,
        "WH": WH,
        "frame_size": tuple(map(int, frame_size)),
        "n_steps": int(n_steps),
    }
    payload = ensure_script(payload)
    if cache_path:
        save_scene(cache_path, payload)
    return payload


def get_scene(
    frame_size: tuple[int, int],
    seed: int | None = None,
    cache_path: str | None = None,
    **kwargs,
) -> Dict:
    """Public entry point returning a generated scene.

    A new :class:`numpy.random.Generator` is created from ``seed`` and passed to
    :func:`generate_scene`.
    """
    rng = np.random.default_rng(seed)
    route_cfg = kwargs.get("route_cfg", RouteGenCfg())
    area_cfg = kwargs.get("area_cfg", AreaGenCfg())
    n_points_range = kwargs.get("n_points_range", (5, 10))
    n_lines_range = kwargs.get("n_lines_range", (1, 3))
    n_polys_range = kwargs.get("n_polys_range", (1, 2))
    n_labels = kwargs.get("n_labels", 5)
    n_steps = kwargs.get("n_steps", 10)
    return generate_scene(
        rng,
        frame_size,
        route_cfg,
        area_cfg,
        n_points_range,
        n_lines_range,
        n_polys_range,
        n_labels,
        n_steps,
        cache_path=cache_path,
    )

