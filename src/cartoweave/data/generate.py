"""Programmatic generators for synthetic scenes and behaviors."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from cartoweave.contracts.solvepack import Scene, Label, Anchor
from cartoweave.config.schema import DataGenerate
from cartoweave.data.sampling.helpers import (
    frame_metrics,
    inset_rect,
    project_to_rect_inset,
)
from cartoweave.data.primitives.polygons import generate_polygon_by_area
from cartoweave.data.sampling.poisson import poisson_disc

__all__ = ["generate_scene"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _polyline_by_length(
    rng: np.random.Generator,
    frame_size: Tuple[float, float],
    L_target: float,
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
    ang = rng.uniform(-math.pi, math.pi)
    total = 0.0
    while total < L_target:
        ang += rng.normal(0.0, angle_sigma)
        step = max(
            min_vertex_spacing,
            rng.normal(segment_len_scale, 0.3 * segment_len_scale),
        )
        q = p + step * np.array([math.cos(ang), math.sin(ang)])
        q = project_to_rect_inset(q, rect)
        if np.linalg.norm(q - p) < min_vertex_spacing:
            ang += math.pi / 2.0
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
        return np.mean(poly, axis=0)
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

    W, H = map(float, gen_cfg.frame_size)
    diag, area_total = frame_metrics((W, H))
    counts = gen_cfg.counts
    spacing = gen_cfg.spacing
    shapes = gen_cfg.shapes
    rt = gen_cfg.route_gen
    ag = gen_cfg.area_gen
    anchors = gen_cfg.anchors

    # Points via simple Poisson sampling inside margin
    if counts.points > 0:
        pts = poisson_disc(
            rng,
            W - 2 * spacing.margin,
            H - 2 * spacing.margin,
            spacing.min_point_dist,
            x0=spacing.margin,
            y0=spacing.margin,
        )
        points = pts[: counts.points]
        if points.shape[0] < counts.points:
            # fallback random sampling if Poisson not enough
            while points.shape[0] < counts.points:
                p = rng.uniform(
                    [spacing.margin, spacing.margin],
                    [W - spacing.margin, H - spacing.margin],
                )
                if not len(points) or np.min(np.linalg.norm(points - p, axis=1)) >= spacing.min_point_dist:
                    points = np.vstack([points, p]) if len(points) else np.array([p])
        points = points[: counts.points]
    else:
        points = np.zeros((0, 2), float)

    # Lines
    lines: List[np.ndarray] = []
    seg_len0 = rt.segment_len_scale * diag
    min_spacing = max(1e-6, rt.min_vertex_spacing_scale * seg_len0)
    inset = spacing.margin + rt.inset_margin_scale * diag
    for _ in range(int(counts.lines)):
        L_target = seg_len0 * rng.uniform(4.0, 6.0)
        line = _polyline_by_length(
            rng,
            (W, H),
            L_target,
            min_spacing,
            inset,
            seg_len0,
            rt.angle_noise_sigma,
        )
        tries = 0
        while line.shape[0] < shapes.line_min_vertices and tries < 5:
            line = _polyline_by_length(
                rng,
                (W, H),
                L_target,
                min_spacing,
                inset,
                seg_len0,
                rt.angle_noise_sigma,
            )
            tries += 1
        lines.append(line)

    # Areas
    areas: List[dict] = []
    area_inset = spacing.margin + ag.inset_margin_scale * diag
    edge_spacing = ag.min_edge_spacing_scale * diag
    for _ in range(int(counts.areas)):
        poly = generate_polygon_by_area(
            rng,
            (W, H),
            0.02 * area_total,
            area_inset,
            edge_spacing,
            (ag.n_vertices_min, ag.n_vertices_max),
        )
        areas.append({"exterior": poly, "holes": []})

    # Build labels and initial positions
    L_total = counts.points + counts.lines + counts.areas
    P0 = np.zeros((L_total, 2), float)
    labels: List[Label] = []
    active0 = np.ones(L_total, dtype=bool)
    idx = 0
    for pt in points:
        anchor = Anchor(mode="xy", xy=(float(pt[0]), float(pt[1])))
        labels.append(Label(id=idx, kind="point", anchor=anchor))
        P0[idx] = pt
        idx += 1
    line_mode = anchors.modes.get("line", "midpoint")
    line_anchor_map = {
        "midpoint": "line_midpoint",
        "centroid": "centroid",
        "projected": "xy",
    }
    for line in lines:
        xy = _line_anchor_xy(line, line_mode, rng)
        amode = line_anchor_map.get(line_mode, "line_midpoint")
        anchor = Anchor(mode=amode, xy=tuple(map(float, xy)) if amode == "xy" else None)
        labels.append(
            Label(id=idx, kind="line", anchor=anchor, polyline=[tuple(map(float, p)) for p in line])
        )
        P0[idx] = xy
        idx += 1
    area_mode = anchors.modes.get("area", "centroid")
    area_anchor_map = {
        "centroid": "centroid",
        "center": "bbox_center",
        "nearest_edge": "xy",
    }
    for area in areas:
        poly = area["exterior"]
        xy = _area_anchor_xy(poly, area_mode, rng)
        amode = area_anchor_map.get(area_mode, "centroid")
        anchor = Anchor(mode=amode, xy=tuple(map(float, xy)) if amode == "xy" else None)
        labels.append(
            Label(id=idx, kind="area", anchor=anchor, polygon=[tuple(map(float, p)) for p in poly])
        )
        P0[idx] = xy
        idx += 1

    scene = Scene(frame_size=(W, H))
    return P0, labels, active0, scene
