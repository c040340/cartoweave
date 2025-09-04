from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import math
import numpy as np

from .layout_utils.geometry import (
    project_point_to_polyline,
    area_anchor_from_centroid_nearest_edge,
)
from .utils.kernels import EPS_DIST


def _as_polyline(obj: Any) -> Optional[np.ndarray]:
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        if arr.ndim == 2 and arr.shape[1] == 4:
            arr2 = np.stack([arr[:, :2], arr[:, 2:]], axis=1)
            return arr2
        if arr.ndim == 1 and arr.shape[0] == 4:
            return arr.reshape(2, 2)
    if isinstance(obj, (list, tuple)):
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
            if lines.ndim == 2 and lines.shape[1] == 4 and 0 <= index < lines.shape[0]:
                seg = lines[index]
                poly = np.array([[seg[0], seg[1]], [seg[2], seg[3]]], float)
        elif isinstance(lines, (list, tuple)) and 0 <= index < len(lines):
            poly = _as_polyline(lines[index])
        if poly is None:
            return (float("nan"), float("nan"), {}) if with_meta else (float("nan"), float("nan"))
        mid = 0.5 * (poly[0] + poly[-1])
        qx, qy, seg_i, t = project_point_to_polyline(mid[0], mid[1], poly)
        meta = {"seg_index": seg_i, "t": t, "polyline": poly}
        return (qx, qy, meta) if with_meta else (qx, qy)

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
