"""Utilities for loading stored scene snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from dataclasses import asdict, is_dataclass

from cartoweave.contracts.solvepack import Scene, Label

__all__ = ["load_snapshot", "save_snapshot"]


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
    return Scene.model_validate(obj)


def _parse_labels(raw: List[Dict[str, Any]]) -> List[Label]:
    out: List[Label] = []
    for lm in raw:
        if isinstance(lm, dict):
            out.append(Label.model_validate(lm))
    return out


def _parse_behaviors(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(b) for b in (raw or [])]


def load_snapshot(path: str) -> Tuple[Scene, np.ndarray, np.ndarray, List[Label], List[Dict[str, Any]]]:
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

    scene_obj = obj.get("scene0") or obj.get("scene")
    if scene_obj is None:
        raise ValueError("snapshot missing 'scene0'")
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


def _to_serializable(obj: Any):
    """Convert objects into JSON-serializable structures."""

    if hasattr(obj, "model_dump"):
        return _to_serializable(obj.model_dump())
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    try:
        return float(obj)
    except Exception:  # pragma: no cover - rare
        return repr(obj)


def save_snapshot(pack, path: str, fmt: str = "json") -> None:
    """Save a SolvePack snapshot to ``path`` in JSON format."""

    if fmt.lower() != "json":
        raise ValueError(f"Unsupported snapshot format: {fmt}")
    from cartoweave.contracts.solvepack import SolvePack  # type: ignore

    obj = {
        "L": pack.L,
        "P0": _to_serializable(np.asarray(pack.P0, dtype=float)),
        "active0": _to_serializable(np.asarray(pack.active0, dtype=bool)),
        "labels0": _to_serializable(pack.labels0),
        "scene0": _to_serializable(pack.scene0),
        "cfg": _to_serializable(pack.cfg),
        "rng_seed": _to_serializable(getattr(pack, "rng_seed", None)),
        "uid": _to_serializable(getattr(pack, "uid", None)),
        "created_at": _to_serializable(getattr(pack, "created_at", None)),
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False))

