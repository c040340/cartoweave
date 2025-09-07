"""Geometry anchor resolver. Maps Label(anchor target/index/mode/xy?) + Scene pools -> anchor coordinate (x,y). Not a force."""
from __future__ import annotations

from typing import Any

import numpy as np

from cartoweave.layout_utils.geometry import (
    poly_centroid,
    project_point_to_polyline,
    project_point_to_segment,
)


def _point_offset(label: Any) -> np.ndarray:
    a = label.anchor
    if a is None:
        return np.zeros(2, float)
    if a.xy is not None:
        return np.asarray(a.xy, float)
    meta = getattr(label, "meta", {}) or {}
    off = meta.get("offset")
    if off is not None:
        return np.asarray(off, float)
    return np.zeros(2, float)


def anchor_position(
    label: Any, scene: Any, P: np.ndarray | None = None  # noqa: N803
) -> np.ndarray:
    """Resolve anchor coordinate for ``label`` in ``scene``.

    Parameters
    ----------
    label:
        Label instance carrying ``anchor`` and ``id``.
    scene:
        Scene or mapping with ``points``, ``lines`` and ``areas`` pools.
    P:
        Optional ``(L,2)`` array of label positions used by certain modes.
    """

    a = getattr(label, "anchor", None)
    if a is None:
        return np.zeros(2, float)

    t = getattr(a, "target", None)
    idx = getattr(a, "index", None)
    mode = getattr(a, "mode", None)

    pts = np.asarray(getattr(scene, "points", []), float)
    lines = getattr(scene, "lines", [])
    areas = getattr(scene, "areas", [])

    if t == "point" and idx is not None and 0 <= idx < len(pts):
        base = np.asarray(pts[idx], float)
        if mode == "offset":
            return base + _point_offset(label)
        return base

    if t == "line" and idx is not None and 0 <= idx < len(lines):
        poly = np.asarray(lines[idx], float)
        if mode == "centroid":
            return poly.mean(axis=0)
        if mode == "projected":
            if P is not None and getattr(label, "id", None) is not None:
                ref = np.asarray(P[int(label.id)], float)
            elif a.xy is not None:
                ref = np.asarray(a.xy, float)
            else:
                ref = poly.mean(axis=0)
            q, _, _ = project_point_to_polyline(ref, poly)
            return np.asarray(q, float)
        # default midpoint
        return 0.5 * (poly[0] + poly[-1])

    if t == "area" and idx is not None and 0 <= idx < len(areas):
        poly = np.asarray(areas[idx], float)
        if mode == "center":
            mins = poly.min(axis=0)
            maxs = poly.max(axis=0)
            return 0.5 * (mins + maxs)
        if mode == "nearest_edge":
            if P is not None and getattr(label, "id", None) is not None:
                ref = np.asarray(P[int(label.id)], float)
            elif a.xy is not None:
                ref = np.asarray(a.xy, float)
            else:
                cx, cy = poly_centroid(poly)
                ref = np.array([cx, cy], float)
            best_q = poly[0]
            best_d2 = float("inf")
            n = poly.shape[0]
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                qx, qy, _ = project_point_to_segment(ref[0], ref[1], x1, y1, x2, y2)
                dx, dy = ref[0] - qx, ref[1] - qy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_q = np.array([qx, qy], float)
            return np.asarray(best_q, float)
        # default centroid
        cx, cy = poly_centroid(poly)
        return np.array([cx, cy], float)

    if t == "free" and a.xy is not None:
        return np.asarray(a.xy, float)

    return np.zeros(2, float)


__all__ = ["anchor_position"]
