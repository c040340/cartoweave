from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cartoweave.contracts.solvepack import SolvePack

from . import config_io, file_io, labels_init
from .build_random import assemble
from .pack_utils import active_mask0_from_stages, steps_to_stages

__all__ = [
    "build_solvepack_from_config",
    "load_solvepack_from_file",
    "build_solvepack_direct",
]


def _payload_from_params(params: Mapping[str, Any], rng: np.random.Generator) -> dict:
    source = params.get("source", "random")
    if source == "file":
        path = params.get("path", "")
        regen = params.get("regen", False)
        if path and Path(path).exists() and not regen:
            payload = file_io.load_scene(path)
        else:
            payload = assemble.generate_payload(params, rng)
            if path:
                file_io.save_scene(path, payload)
    else:
        payload = assemble.generate_payload(params, rng)
    return payload


def _ensure_payload(payload: dict, params: Mapping[str, Any], rng: np.random.Generator) -> dict:
    geom = {
        "points": payload.get("points", np.zeros((0, 2), float)),
        "lines": payload.get("lines", []),
        "areas": payload.get("areas", []),
        "frame_size": payload.get(
            "frame_size", (params["frame"]["width"], params["frame"]["height"])
        ),
    }
    if "labels" not in payload:
        n_labels = params["counts"]["labels"]
        anchors = params.get("anchors", {})
        payload["labels"] = assemble.assign_labels(
            geom,
            n_labels,
            anchors.get("policy", "auto"),
            anchors.get("modes", {}),
            rng,
        )
    if "WH" not in payload:
        payload["WH"] = assemble.measure_WH(len(payload["labels"]), rng)
    return payload


def _build_solvepack(payload: dict, steps_cfg: Mapping[str, Any], solver_cfg: dict | None, rng: np.random.Generator) -> SolvePack:
    n_labels = len(payload["labels"])
    stages = steps_to_stages(steps_cfg, n_labels)
    active_mask0 = active_mask0_from_stages(stages, n_labels)
    p0 = labels_init.compute_P0(payload, rng)
    return SolvePack(
        n_labels,
        p0,
        active_mask0,
        scene=payload,
        cfg=solver_cfg or {},
        stages=stages,
        passes=["schedule", "capture"],
    )


def build_solvepack_from_config(config: str | Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None, seed: int | None = None) -> SolvePack:
    merged = config_io.load_and_merge(config, overrides or {})
    params = config_io.extract_data_params(merged)
    rng = np.random.default_rng(seed)
    payload = _payload_from_params(params, rng)
    payload = _ensure_payload(payload, params, rng)
    return _build_solvepack(payload, params.get("steps", {}), merged.get("solver_cfg"), rng)


def load_solvepack_from_file(path: str, *, solver_cfg: dict | None = None, seed: int | None = None) -> SolvePack:
    rng = np.random.default_rng(seed)
    payload = file_io.load_scene(path)
    params = {
        "frame": {"width": payload.get("frame_size", (1, 1))[0], "height": payload.get("frame_size", (1, 1))[1]},
        "counts": {"labels": len(payload.get("labels", []))},
        "anchors": {"policy": "auto", "modes": {}},
    }
    payload = _ensure_payload(payload, params, rng)
    steps_cfg = payload.get("steps", {"kind": "none", "steps": None})
    return _build_solvepack(payload, steps_cfg, solver_cfg, rng)


def build_solvepack_direct(*, frame_size: tuple[int, int], n_labels: int, n_points: int = 0, n_lines: int = 0, n_areas: int = 0, route_gen: dict | None = None, area_gen: dict | None = None, anchors_policy: dict | str | None = None, anchor_modes: dict | None = None, steps: dict | str | None = None, seed: int | None = None, solver_cfg: dict | None = None) -> SolvePack:
    route_defaults = config_io._route_defaults()
    area_defaults = config_io._area_defaults()
    rgen = route_defaults | (route_gen or {})
    agen = area_defaults | (area_gen or {})
    params = {
        "source": "random",
        "frame": {"width": int(frame_size[0]), "height": int(frame_size[1])},
        "counts": {
            "labels": int(n_labels),
            "points": int(n_points),
            "lines": int(n_lines),
            "areas": int(n_areas),
        },
        "random": {
            "route_gen": rgen,
            "area_gen": agen,
        },
        "anchors": {
            "policy": anchors_policy if anchors_policy is not None else "auto",
            "modes": anchor_modes or {},
        },
    }
    if isinstance(steps, str):
        params["steps"] = {"kind": steps, "steps": None}
    elif steps is not None:
        params["steps"] = dict(steps)
    rng = np.random.default_rng(seed)
    payload = assemble.generate_payload(params, rng)
    payload = _ensure_payload(payload, params, rng)
    steps_cfg = params.get("steps", {"kind": "none", "steps": None})
    return _build_solvepack(payload, steps_cfg, solver_cfg, rng)
