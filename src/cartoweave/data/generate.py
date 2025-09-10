"""Programmatic generators for synthetic scenes."""
from __future__ import annotations

import math

import numpy as np

from cartoweave.config.schema import DataGenerate
from cartoweave.contracts.solvepack import Anchor, Label, Scene
from cartoweave.data.build_random.areas import generate_areas
from cartoweave.data.primitives.polygons import generate_polygon_by_area
from cartoweave.data.sampling.helpers import (
    frame_metrics,
    inset_rect,
    project_to_rect_inset,
)
from cartoweave.data.sampling.poisson import poisson_disc
from cartoweave.data.textblock import (
    load_font,
    measure_text_block,
    random_text_lines,
)
from cartoweave.compute.geometry import (
    polyline_uniform_arc_centroid,
    project_point_to_polyline,
)

__all__ = ["generate_scene"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polyline_by_length(
    rng: np.random.Generator,
    frame_size: tuple[float, float],
    length_target: float,
    min_vertex_spacing: float,
    inset_margin: float,
    segment_len_scale: float,
    angle_sigma: float,
) -> np.ndarray:
    """Generate a random polyline with configurable parameters."""
    rect = inset_rect(frame_size, inset_margin)
    x0, y0, x1, y1 = rect
    p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
    pts = [p]
    total = 0.0
    ang = rng.uniform(-math.pi, math.pi)
    while total < length_target:
        ang += rng.normal(0, angle_sigma * math.pi)
        step = max(
            min_vertex_spacing,
            rng.normal(min_vertex_spacing, 0.3 * min_vertex_spacing),
        )
        q = p + step * np.array([math.cos(ang), math.sin(ang)])
        q = project_to_rect_inset(q, rect)
        if np.linalg.norm(q - p) < min_vertex_spacing:
            #ang += math.pi / 2.0
            q = project_to_rect_inset(
                p + step * np.array([math.cos(ang), math.sin(ang)]), rect
            )
        total += float(np.linalg.norm(q - p))
        pts.append(q)
        p = q
    return np.asarray(pts, dtype=float)


def _polyline_midpoint(poly: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = float(np.sum(d))
    if total == 0.0:
        return poly[0]
    half = 0.5 * total
    cum = np.cumsum(d)
    idx = int(np.searchsorted(cum, half))
    prev = poly[idx]
    nxt = poly[min(idx + 1, poly.shape[0] - 1)]
    seg_len = d[idx] if idx < len(d) else 1.0
    t = (half - (cum[idx - 1] if idx > 0 else 0.0)) / seg_len
    return (1 - t) * prev + t * nxt


def _line_anchor_xy(poly: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "midpoint":
        return _polyline_midpoint(poly)
    if mode == "centroid":
        c = polyline_uniform_arc_centroid(poly, step_len=4.0)
        q, _, _ = project_point_to_polyline(c, poly)
        return q
    if mode == "projected":
        # random point along polyline
        d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        total = float(np.sum(d))
        if total == 0.0:
            return poly[0]
        t = rng.uniform(0.0, total)
        cum = np.cumsum(d)
        idx = int(np.searchsorted(cum, t))
        prev = poly[idx]
        nxt = poly[min(idx + 1, poly.shape[0] - 1)]
        seg_len = d[idx] if idx < len(d) else 1.0
        u = (t - (cum[idx - 1] if idx > 0 else 0.0)) / seg_len
        return (1 - u) * prev + u * nxt
    return poly[0]


def _area_anchor_xy(poly: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "centroid":
        return np.mean(poly, axis=0)
    if mode == "center":
        xmin, ymin = np.min(poly, axis=0)
        xmax, ymax = np.max(poly, axis=0)
        return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    if mode == "nearest_edge":
        c = np.mean(poly, axis=0)
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        mids = edges.mean(axis=1)
        d = np.linalg.norm(mids - c, axis=1)
        return mids[int(np.argmin(d))]
    return poly[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_scene(gen_cfg: DataGenerate, rng: np.random.Generator):
    """Generate geometry, labels and initial positions from ``gen_cfg``."""
    backend = getattr(gen_cfg, "backend", "build_random")
    w, h = map(float, gen_cfg.frame_size)
    diag, area_total = frame_metrics((w, h))
    counts = gen_cfg.counts
    spacing = gen_cfg.spacing
    shapes = gen_cfg.shapes
    rt = gen_cfg.route_gen
    ag = gen_cfg.area_gen
    anchors = gen_cfg.anchors

    # Points via simple Poisson sampling inside margin
    if counts.points > 0:
        pts = []
        for i in range(counts.points):
            x = rng.uniform(0.0, w - 2 * spacing.margin)
            y = rng.uniform(0.0, h - 2 * spacing.margin)
            pts.append([x, y])
        pts = np.array(pts)
        points = pts[: counts.points]
        if points.shape[0] < counts.points:
            # fallback random sampling if Poisson not enough
            while points.shape[0] < counts.points:
                p = rng.uniform(
                    [spacing.margin, spacing.margin],
                    [w - spacing.margin, h - spacing.margin],
                )
                if not len(points) or np.min(np.linalg.norm(points - p, axis=1)) >= spacing.min_point_dist:
                    points = np.vstack([points, p]) if len(points) else np.array([p])
        points = points[: counts.points]
    else:
        points = np.zeros((0, 2), float)

    # Lines
    lines: list[np.ndarray] = []
    seg_len0 = rt.segment_len_scale * diag
    min_spacing = max(1e-6, rt.min_vertex_spacing_scale * seg_len0)
    inset = spacing.margin + rt.inset_margin_scale * diag
    for _ in range(int(counts.lines)):
        length_target = seg_len0 * rng.uniform(0.8, 1.2)
        line = _polyline_by_length(
            rng,
            (w, h),
            length_target,
            min_spacing,
            inset,
            seg_len0,
            rt.angle_noise_sigma,
        )
        tries = 0
        while line.shape[0] < shapes.line_min_vertices and tries < 5:
            line = _polyline_by_length(
                rng,
                (w, h),
                length_target,
                min_spacing,
                inset,
                seg_len0,
                rt.angle_noise_sigma,
            )
            tries += 1
        lines.append(line)

    # Areas
    areas: list[np.ndarray] = []
    if counts.areas > 0:
        if backend == "build_random":
            # 适配 build_random.areas.generate_areas(params, rng) 的入参风格
            params = {
                "frame": {"width": int(w), "height": int(h)},
                "counts": {"areas": int(counts.areas)},
                "random": {
                    "area_gen": {
                        "inset_margin_scale": float(ag.inset_margin_scale),
                        "min_edge_spacing_scale": float(ag.min_edge_spacing_scale),
                        "n_vertices_min": int(ag.n_vertices_min),
                        "n_vertices_max": int(ag.n_vertices_max),
                    }
                },
            }
            polys = generate_areas(params, rng)
            areas = [np.asarray(p, float) for p in polys]
        else:
            # 保留原 primitives 路线作为备选
            area_inset = spacing.margin + ag.inset_margin_scale * diag
            edge_spacing = ag.min_edge_spacing_scale * diag
            for _ in range(int(counts.areas)):
                poly = generate_polygon_by_area(
                    rng,
                    (w, h),
                    0.02 * area_total,  # 原来的 2% 目标面积
                    area_inset,
                    edge_spacing,
                    (ag.n_vertices_min, ag.n_vertices_max),
                )
                areas.append(poly)

    # Convert geometry to scene pools
    points_list = [tuple(map(float, p)) for p in points]
    lines_list = [[tuple(map(float, p)) for p in line] for line in lines]
    areas_list = [[tuple(map(float, p)) for p in poly] for poly in areas]

    scene = Scene(
        frame_size=(w, h),
        points=points_list,
        lines=lines_list,
        areas=areas_list,
    )

    # Determine label count
    label_count = (
        gen_cfg.labels
        if gen_cfg.labels is not None
        else counts.points + counts.lines + counts.areas
    )

    p0 = np.zeros((label_count, 2), float)
    labels: list[Label] = []

    txt_cfg = gen_cfg.text
    font = load_font(txt_cfg.font.path, int(txt_cfg.font.size))
    len_min, len_max = map(int, txt_cfg.len_range)
    spacing = int(txt_cfg.line_spacing_px)
    padx = int(txt_cfg.padding_px.x)
    pady = int(txt_cfg.padding_px.y)

    pool_sizes = {
        "point": len(points_list),
        "line": len(lines_list),
        "area": len(areas_list),
    }

    def _sample_kind() -> str | None:
        if gen_cfg.label_mix:
            weights = {k: v for k, v in gen_cfg.label_mix.items() if v > 0}
        else:
            weights = {k: pool_sizes[k] for k in pool_sizes}
        available = [k for k in ("point", "line", "area") if pool_sizes[k] > 0 and weights.get(k, 0) > 0]
        if not available:
            return None
        probs = np.array([weights[k] for k in available], float)
        probs /= probs.sum()
        return str(rng.choice(available, p=probs))

    # ensure at least one label for each available geometry kind
    available_kinds = [k for k in ("point", "line", "area") if pool_sizes[k] > 0]
    label_kinds: list[str | None] = []
    if label_count >= len(available_kinds):
        label_kinds.extend(available_kinds)
        for _ in range(label_count - len(label_kinds)):
            label_kinds.append(_sample_kind())
        rng.shuffle(label_kinds)
    else:
        for _ in range(label_count):
            label_kinds.append(_sample_kind())

    rr_next = {"point": 0, "line": 0, "area": 0}

    for i, kind in enumerate(label_kinds):
        if kind is None:
            xy = rng.uniform([0.0, 0.0], [w, h])
            anchor = Anchor(target="free", mode="xy", xy=(float(xy[0]), float(xy[1])))
            label_kind = "point"
        else:
            label_kind = kind
            size = pool_sizes.get(kind, 0)
            if size == 0:
                avail = [k for k in ("point", "line", "area") if pool_sizes[k] > 0]
                if not avail:
                    xy = rng.uniform([0.0, 0.0], [w, h])
                    anchor = Anchor(target="free", mode="xy", xy=(float(xy[0]), float(xy[1])))
                    label_kind = "point"
                else:
                    kind = str(rng.choice(avail))
                    label_kind = kind
                    size = pool_sizes[kind]
            if size > 0:
                if anchors.policy == "round_robin":
                    idx = rr_next[kind] % size
                    rr_next[kind] += 1
                elif anchors.policy == "random":
                    idx = int(rng.integers(0, size))
                else:  # fixed
                    idx = 0
                if kind == "point":
                    xy = np.asarray(points_list[idx], float)
                    anchor = Anchor(target="point", index=idx, mode="exact")
                elif kind == "line":
                    mode = anchors.modes.get("line", "midpoint")
                    xy = _line_anchor_xy(np.asarray(lines_list[idx], float), mode, rng)
                    anchor = Anchor(target="line", index=idx, mode=mode)
                else:  # area
                    mode = anchors.modes.get("area", "centroid")
                    xy = _area_anchor_xy(np.asarray(areas_list[idx], float), mode, rng)
                    anchor = Anchor(target="area", index=idx, mode=mode)
        p0[i] = xy
        lines = random_text_lines(rng, (len_min, len_max))
        W, H = measure_text_block(lines, font, spacing, padx, pady)
        labels.append(
            Label(
                id=i,
                kind=label_kind,
                anchor=anchor,
                WH=(float(W), float(H)),
                meta={"mode": "rectangular", "text_lines": lines},
            )
        )

    return p0, labels, scene
