"""Top-level entry points for building :class:`SolvePack` objects."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
from typing import Dict, Mapping, Any, List
import json
import yaml

import numpy as np

from cartoweave.contracts.solvepack import (
    SolvePack,
    validate,
    Scene,
    LabelState, Behavior,
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
    """Construct a :class:`SolvePack` from a merged configuration or YAML path (v2 only)."""
    # --- 0) Load cfg dict ---
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        cfg = dict(config or {})

    # --- 1) Prepare defaults up-front (so generators see them) ---
    user_behavior = dict(cfg.get("behavior", {}) or {})
    behavior_cfg: Dict[str, Any] = {
        "place_on_first_activation": True,
        "snap_on_kind_change": False,
        "default_WH": {"point": [8, 8], "line": [12, 6], "area": [40, 30]},
        "anchor_policy": "round_robin",
    }
    behavior_cfg.update(user_behavior)

    user_solver = dict(cfg.get("solver", {}) or {})
    solver_cfg: Dict[str, Any] = {"gtol": 1e-6, "ftol": 1e-9, "xtol": 1e-9}
    solver_cfg.update(user_solver)

    # --- 2) Branch: generate / load ---
    data_cfg = dict(cfg.get("data", {}) or {})
    source = data_cfg.get("source")
    if source == "generate":
        g = dict(data_cfg.get("generate", {}) or {})
        frame = tuple(g.get("frame_size", (1920.0, 1080.0)))
        seed = int(g.get("seed", seed or 0))

        num_points = int(g.get("num_points", 0))
        num_lines  = int(g.get("num_lines", 0))
        num_areas  = int(g.get("num_areas", 0))
        num_steps  = int(g.get("num_steps", 1))
        scene = generate_scene(num_points, num_lines, num_areas, frame_size=frame, seed=seed)

        nl = g.get("num_labels")
        N = int(nl) if nl is not None else max(1, num_points + num_lines + num_areas)

        P0, active0, labels0 = generate_labels(N, scene, behavior_cfg)
        behaviors = generate_behaviors(
            N,
            num_steps,
            scene,
            behavior_cfg=behavior_cfg,
            policy=behavior_cfg.get("anchor_policy", "round_robin"),
            seed=seed,
        )
    elif source == "load":
        load_cfg = dict(data_cfg.get("load", {}) or {})
        path = load_cfg.get("path")
        if not path:
            raise ValueError("data.load.path is required for source=load")
        scene, P0, active0, labels0, behaviors = load_snapshot(str(path))
        N = int(P0.shape[0])
    else:
        raise ValueError("data.source must be 'generate' or 'load'")

    # --- 3) Ensure scene/labels consistency (your existing helper) ---
    _ensure_scene_labels(scene, labels0)

    # --- 4) Normalize behaviors to dataclasses (BehaviorOp/Behavior) ---
    norm_behaviors: List[Behavior] = []
    for b in behaviors:
        if isinstance(b, Behavior):
            norm_behaviors.append(b)
            continue
        # assume dict-like
        ops_dict = dict(b.get("ops", {}) if isinstance(b, dict) else {})
        bop = BehaviorOp(
            activate=list(ops_dict.get("activate", [])),
            deactivate=list(ops_dict.get("deactivate", [])),
            mutate=list(ops_dict.get("mutate", [])),
        )
        norm_behaviors.append(
            Behavior(
                iters=int(b.get("iters", 1)) if isinstance(b, dict) else 1,
                ops=bop,
                solver=(b.get("solver", "lbfgs") if isinstance(b, dict) else "lbfgs"),
                params=dict(b.get("params", {}) if isinstance(b, dict) else {}),
            )
        )

    # --- 5) Build pack cfg (only what SolvePack needs) ---
    pack_cfg: Dict[str, Any] = copy.deepcopy(cfg)
    pack_cfg["behavior"] = behavior_cfg
    pack_cfg["solver"] = solver_cfg
    pack_cfg["behaviors"] = norm_behaviors

    # --- 6) Construct v2 SolvePack and validate ---
    pack = SolvePack(N=N, P0=P0, active0=active0, labels0=labels0, scene0=scene, cfg=pack_cfg)
    validate(pack)
    return pack


# Legacy helpers kept for backwards compatibility --------------------------------


def load_solvepack_from_file(*args, **kwargs):  # pragma: no cover - legacy API
    raise NotImplementedError("Use build_solvepack_from_config in v2")


def build_solvepack_direct(*args, **kwargs):  # pragma: no cover - legacy API
    raise NotImplementedError("Use build_solvepack_from_config in v2")

