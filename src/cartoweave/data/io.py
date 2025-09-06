"""Utilities for loading stored scene snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from cartoweave.contracts.solvepack import (
    Scene,
    LabelState,
    AnchorSpec,
    Behavior,
    BehaviorOp,
)

__all__ = ["load_snapshot"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz(path: Path) -> Dict[str, Any]:  # pragma: no cover - rarely used
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        out[k] = data[k].item() if data[k].dtype == object else data[k]
    return out


def _parse_scene(obj: Dict[str, Any]) -> Scene:
    pts = np.asarray(obj.get("points", []), dtype=float)
    lines = [np.asarray(l, float) for l in obj.get("lines", [])]
    areas = []
    for a in obj.get("areas", []):
        exterior = np.asarray(a.get("exterior", []), float)
        holes = [np.asarray(h, float) for h in a.get("holes", [])]
        areas.append({"exterior": exterior, "holes": holes})
    frame_size = tuple(obj.get("frame_size", [1920.0, 1080.0]))
    labels = obj.get("labels", [])
    WH = np.asarray(obj.get("WH", np.zeros((len(labels), 2))), float)
    return Scene(points=pts, lines=lines, areas=areas, frame_size=frame_size, labels=labels, WH=WH)


def _parse_labels(raw: List[Dict[str, Any]]) -> List[LabelState]:
    out: List[LabelState] = []
    for lm in raw:
        anc_raw = lm.get("anchor")
        anc = None
        if isinstance(anc_raw, dict):
            anc = AnchorSpec(kind=anc_raw.get("kind"), index=int(anc_raw.get("index", 0)), t=anc_raw.get("t"))
        WH = np.asarray(lm.get("WH", [0, 0]), float)
        out.append(LabelState(kind=lm.get("kind", "none"), WH=WH, anchor=anc, meta=lm.get("meta", {})))
    return out


def _parse_behaviors(raw: List[Dict[str, Any]]) -> List[Behavior]:
    out: List[Behavior] = []
    for b in raw or []:
        ops = b.get("ops", {}) or {}
        mut = []
        for m in ops.get("mutate", []):
            mut.append({"id": int(m.get("id")), "set": m.get("set", {})})
        ops_norm = {
            "activate": [int(i) for i in ops.get("activate", [])],
            "deactivate": [int(i) for i in ops.get("deactivate", [])],
            "mutate": mut,
        }
        out.append(
            Behavior(
                iters=int(b.get("iters", 0)),
                ops=ops_norm,
                solver=b.get("solver", "lbfgs"),
                params=b.get("params", {}),
            )
        )
    return out


def load_snapshot(path: str) -> Tuple[Scene, np.ndarray, np.ndarray, List[LabelState], List[Behavior]]:
    """Load a snapshot from ``path`` which may be ``.json`` or ``.npz``."""

    p = Path(path)
    if not p.exists():
        raise ValueError(f"snapshot path not found: {path}")
    if p.suffix.lower() == ".json":
        obj = _load_json(p)
    elif p.suffix.lower() == ".npz":  # pragma: no cover - seldom used
        obj = _load_npz(p)
    else:
        raise ValueError(f"unsupported snapshot format: {path}")

    scene_obj = obj.get("scene")
    if scene_obj is None:
        raise ValueError("snapshot missing 'scene'")
    scene = _parse_scene(scene_obj)

    P0 = np.asarray(obj.get("P0"), float)
    active0 = np.asarray(obj.get("active0"), bool)
    labels0 = _parse_labels(obj.get("labels0", []))
    behaviors = _parse_behaviors(obj.get("behaviors", []))

    if P0.ndim != 2 or P0.shape[1] != 2:
        raise ValueError("P0 must have shape (N,2)")
    if active0.shape != (P0.shape[0],):
        raise ValueError("active0 shape mismatch")
    if len(labels0) != P0.shape[0]:
        raise ValueError("labels0 length mismatch")

    return scene, P0, active0, labels0, behaviors

