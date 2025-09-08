from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import math
import numpy as np

# geometry utilities moved from layout_utils to compute.geometry
from cartoweave.compute.geometry import (
    project_point_to_polyline,
    polyline_uniform_arc_centroid,
    area_anchor_from_centroid_nearest_edge,
)
from .utils.kernels import EPS_DIST


def _as_polyline(obj: Any) -> Optional[np.ndarray]:
    """Return ``obj`` as a ``(N,2)`` float array if possible."""
    if isinstance(obj, (np.ndarray, list, tuple)):
        arr = np.asarray(obj, float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    return None


def anchor_xy(kind: str, index: int, data: Dict[str, Any], frame: Tuple[float, float], *, with_meta: bool = False):
    """Dispatch to the appropriate anchor computation.

    Parameters
    ----------
    kind/index:
        Anchor kind and element index.
    data:
        Mapping containing ``points``, ``lines`` and ``areas`` entries as needed.
    frame:
        Frame size ``(W, H)`` (currently unused but kept for compatibility).
    with_meta:
        When ``True`` return ``(x, y, meta)`` with extra information.
    """
    k = (kind or "").lower()
    if k == "point":
        pts = np.asarray(data.get("points"), float)
        if 0 <= index < len(pts):
            qx, qy = float(pts[index, 0]), float(pts[index, 1])
            return (qx, qy, {}) if with_meta else (qx, qy)
        return (float("nan"), float("nan"), {}) if with_meta else (float("nan"), float("nan"))

    if k == "line":
        lines = data.get("lines")
        poly = None
        if isinstance(lines, np.ndarray):
            if lines.ndim == 3 and lines.shape[2] == 2 and 0 <= index < lines.shape[0]:
                poly = lines[index]
        elif isinstance(lines, (list, tuple)) and 0 <= index < len(lines):
            item = lines[index]
            if isinstance(item, dict):
                poly = _as_polyline(item.get("polyline"))
            else:
                poly = _as_polyline(item)
        if poly is None:
            return (float("nan"), float("nan"), {}) if with_meta else (float("nan"), float("nan"))

        C = polyline_uniform_arc_centroid(poly, step_len=4.0)
        Q, tau, seg_i = project_point_to_polyline(C, poly)
        base_n = np.array([-tau[1], tau[0]], float)
        side = np.sign(np.dot(C - Q, base_n))
        if side == 0.0:
            side = 1.0
        n = base_n * side
        delta = 0.0
        anchor = Q + delta * n
        meta = {
            "seg_index": seg_i,
            "polyline": poly,
            "tangent": (float(tau[0]), float(tau[1])),
            "normal": (float(n[0]), float(n[1])),
        }
        ax, ay = float(anchor[0]), float(anchor[1])
        return (ax, ay, meta) if with_meta else (ax, ay)

    if k == "area":
        areas = data.get("areas")
        poly = None
        if isinstance(areas, (list, tuple)) and 0 <= index < len(areas):
            ar = areas[index]
            if isinstance(ar, dict):
                poly = np.asarray(ar.get("polygon"), float)
            else:
                poly = np.asarray(ar, float)
        if poly is None:
            return (float("nan"), float("nan"), {}) if with_meta else (float("nan"), float("nan"))
        info = area_anchor_from_centroid_nearest_edge(poly)
        meta = dict(info)
        qx = meta.pop("qx")
        qy = meta.pop("qy")
        return (qx, qy, meta) if with_meta else (qx, qy)

    return (float("nan"), float("nan"), {}) if with_meta else (float("nan"), float("nan"))


def init_position(
    kind: str,
    anchor: Tuple[float, float],
    frame: Tuple[float, float],
    prev: Optional[Tuple[float, float]] = None,
    *,
    mode: Optional[str] = None,
    locked: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return initial label position according to anchor/previous state."""
    if prev is not None and not np.any(np.isnan(prev)):
        return np.asarray(prev, float)
    ax, ay = float(anchor[0]), float(anchor[1])
    if (mode or "").lower() == "circle" or locked:
        return np.array([ax, ay], float)
    cx = 0.5 * frame[0]
    cy = 0.5 * frame[1]
    nudge = 3.0
    if (kind or "").lower() == "point":
        vx = cx - ax
        vy = cy - ay
        L = math.hypot(vx, vy)
        if L <= EPS_DIST:
            vx, vy = 1.0, 0.0
            L = 1.0
        vx /= L
        vy /= L
        return np.array([ax + nudge * vx, ay + nudge * vy], float)
    if (kind or "").lower() == "line":
        if meta is not None and "normal" in meta and "tangent" in meta:
            nx, ny = meta["normal"]
            tx, ty = meta["tangent"]
        else:
            poly = None if meta is None else meta.get("polyline")
            seg_i = 0 if meta is None else int(meta.get("seg_index", 0))
            if poly is not None and poly.shape[0] >= 2:
                x1, y1 = poly[seg_i]
                x2, y2 = poly[(seg_i + 1) % poly.shape[0]]
                dx = x2 - x1
                dy = y2 - y1
                L = math.hypot(dx, dy)
                if L <= EPS_DIST:
                    tx, ty = 1.0, 0.0
                else:
                    tx, ty = dx / L, dy / L
                nx, ny = -ty, tx
                if (cx - ax) * nx + (cy - ay) * ny < 0:
                    nx, ny = -nx, -ny
            else:
                nx, ny = 0.0, 1.0
        return np.array([ax + nudge * nx, ay + nudge * ny], float)
    if (kind or "").lower() == "area":
        nx, ny = (0.0, 1.0)
        if meta is not None:
            nx, ny = meta.get("normal_in", (0.0, 1.0))
        return np.array([ax + nudge * nx, ay + nudge * ny], float)
    return np.array([ax, ay], float)


__all__ = ["anchor_xy", "init_position"]
