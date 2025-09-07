from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .types import VPSources


def _ensure_points(points_raw: Any) -> np.ndarray:
    pts = np.asarray(points_raw, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points shape {pts.shape} != (M,2)")
    if not np.isfinite(pts).all():
        raise ValueError("points contain non-finite values")
    return pts


def _ensure_lines(lines_raw: Any) -> List[np.ndarray]:
    lines: List[np.ndarray] = []
    for i, ln in enumerate(lines_raw or []):
        arr = np.asarray(ln, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            raise ValueError(f"lines[{i}] shape {arr.shape} invalid")
        if not np.isfinite(arr).all():
            raise ValueError(f"lines[{i}] contain non-finite values")
        lines.append(arr)
    return lines


def _ensure_areas(scene: Dict[str, Any]) -> List[dict]:
    areas: List[dict] = []
    for i, poly in enumerate(scene.get("areas", []) or []):
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            raise ValueError(f"areas[{i}] polygon shape {arr.shape} invalid")
        if not np.isfinite(arr).all():
            raise ValueError(f"areas[{i}] polygon has non-finite values")
        areas.append({"kind": "poly", "xy": arr})
    for i, circ in enumerate(scene.get("circles", []) or []):
        if isinstance(circ, dict):
            xy = np.asarray(circ.get("xy"), dtype=float)
            r = float(circ.get("r", np.nan))
        else:
            if len(circ) != 3:
                raise ValueError(f"circles[{i}] must have (x,y,r)")
            xy = np.asarray(circ[:2], dtype=float)
            r = float(circ[2])
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(r):
            raise ValueError(f"circles[{i}] invalid or non-finite")
        areas.append({"kind": "circle", "xy": xy, "r": r})
    return areas


def make_sources_from_scene(scene: Dict[str, Any]) -> VPSources:
    """Build :class:`VPSources` from a raw scene dictionary.

    Parameters
    ----------
    scene:
        Mapping containing ``points``, ``lines``, ``areas`` and optional
        ``circles`` along with ``frame_size``.
    """

    pts = _ensure_points(scene.get("points", []))
    lines = _ensure_lines(scene.get("lines", []))
    areas = _ensure_areas(scene)

    frame_size_raw = scene.get("frame_size")
    if frame_size_raw is None or len(frame_size_raw) != 2:
        raise ValueError("scene.frame_size must be length-2")
    W, H = frame_size_raw
    if not (np.isfinite(W) and np.isfinite(H)):
        raise ValueError("frame_size must be finite")
    frame_size = (int(W), int(H))

    sources = VPSources(points=pts, lines=lines, areas=areas, frame_size=frame_size)
    sources.validate()
    return sources


__all__ = ["make_sources_from_scene"]
