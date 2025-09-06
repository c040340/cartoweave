"""Top-level entry points for building :class:`SolvePack` objects."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Any
import json
import yaml

import numpy as np

from cartoweave.contracts.solvepack import (
    SolvePack,
    validate,
    Scene,
    LabelState,
)

from .generate import generate_scene, generate_labels, generate_behaviors
from .io import load_snapshot

__all__ = ["build_solvepack_from_config"]


# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------


def _ensure_scene_labels(scene: Scene, labels0: list[LabelState]) -> None:
    scene.labels = [
        {"kind": lbl.kind, "anchor": asdict(lbl.anchor) if lbl.anchor else None, "meta": lbl.meta}
        for lbl in labels0
    ]
    scene.WH = np.stack([lbl.WH for lbl in labels0], axis=0) if labels0 else np.zeros((0, 2))


def build_solvepack_from_config(config: Mapping[str, Any] | str, seed: int | None = None) -> SolvePack:
    """Construct a :class:`SolvePack` from a merged configuration or YAML path."""

    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
    else:
        cfg = dict(config)

    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source")

    if source == "generate":
        g = data_cfg.get("generate", {})
        frame = tuple(g.get("frame_size", [1920.0, 1080.0]))
        seed = int(g.get("seed", seed or 0))
        scene = generate_scene(
            int(g.get("num_points", 0)),
            int(g.get("num_lines", 0)),
            int(g.get("num_areas", 0)),
            frame_size=frame,
            seed=seed,
        )
        nl = g.get("num_labels")
        if nl is None:
            N = max(1, g.get("num_points", 0) + g.get("num_lines", 0) + g.get("num_areas", 0))
        else:
            N = int(nl)
        P0, active0, labels0 = generate_labels(N, scene, cfg.get("behavior"))
        behaviors = generate_behaviors(N, int(g.get("num_steps", 1)), scene, cfg.get("behavior"), "round_robin", seed)
    elif source == "load":
        path = data_cfg.get("load", {}).get("path")
        scene, P0, active0, labels0, behaviors = load_snapshot(str(path))
        N = P0.shape[0]
    else:
        raise ValueError("data.source must be 'generate' or 'load'")

    _ensure_scene_labels(scene, labels0)

    cfg.setdefault(
        "behavior",
        {
            "place_on_first_activation": True,
            "snap_on_kind_change": False,
            "default_WH": {"point": [8, 8], "line": [12, 6], "area": [40, 30]},
            "anchor_policy": "round_robin",
        },
    )
    cfg.setdefault("solver", {"gtol": 1e-6, "ftol": 1e-9, "xtol": 1e-9})
    cfg["behaviors"] = behaviors

    pack = SolvePack(N=N, P0=P0, active0=active0, labels0=labels0, scene0=scene, cfg=cfg)
    validate(pack)
    return pack


# Legacy helpers kept for backwards compatibility --------------------------------


def load_solvepack_from_file(*args, **kwargs):  # pragma: no cover - legacy API
    raise NotImplementedError("Use build_solvepack_from_config in v2")


def build_solvepack_direct(*args, **kwargs):  # pragma: no cover - legacy API
    raise NotImplementedError("Use build_solvepack_from_config in v2")

