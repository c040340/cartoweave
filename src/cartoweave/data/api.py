from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cartoweave.contracts.solvepack import SolvePack

from . import config_io, file_io, labels_init
from .build_random import assemble
from .pack_utils import steps_to_behaviors

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


def _fail_if_legacy(obj: Any) -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k in {"stages", "mask_override", "active_mask0_from_stages"}:
                raise ValueError(
                    "Legacy config detected: migrate to behavior-based pipeline (activate/deactivate/mutate)."
                )
            _fail_if_legacy(v)
    elif isinstance(obj, list):
        for v in obj:
            _fail_if_legacy(v)


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


def _validate_behaviors(behaviors: list[dict], n_labels: int) -> list[dict]:
    out: list[dict] = []
    for b in behaviors:
        if not isinstance(b, Mapping):
            raise ValueError("behavior must be dict")
        ops = b.get("ops", {}) or {}
        act = [int(i) for i in ops.get("activate", [])]
        deact = [int(i) for i in ops.get("deactivate", [])]
        mut = []
        for m in ops.get("mutate", []):
            mid = int(m.get("id"))
            if not (0 <= mid < n_labels):
                raise IndexError(f"mutate.id {mid} out of range")
            mut.append({"id": mid, "set": m.get("set", {})})
        for idx in act + deact:
            if not (0 <= idx < n_labels):
                raise IndexError(f"index {idx} out of range")
        out.append(
            {
                "iters": int(b.get("iters", 0)),
                "ops": {"activate": act, "deactivate": deact, "mutate": mut},
                "solver": b.get("solver", "lbfgs"),
                "params": b.get("params", {}),
            }
        )
    return out


def _build_solvepack(payload: dict, steps_cfg: Mapping[str, Any] | list | None, solver_cfg: dict | None, rng: np.random.Generator) -> SolvePack:
    n_labels = len(payload["labels"])
    behaviors: list[dict] = []
    if steps_cfg:
        if isinstance(steps_cfg, list):
            behaviors = steps_cfg
        elif isinstance(steps_cfg, Mapping) and steps_cfg.get("behaviors") is not None:
            behaviors = steps_cfg.get("behaviors", [])
        else:
            behaviors = steps_to_behaviors(dict(steps_cfg), n_labels)
    behaviors = _validate_behaviors(behaviors, n_labels)

    active_mask0 = np.zeros(n_labels, dtype=bool)
    p0 = labels_init.compute_P0(payload, rng)
    sp = SolvePack(
        n_labels,
        p0,
        active_mask0,
        scene=payload,
        cfg=solver_cfg or {},
        passes=["schedule", "capture"],
    )
    sp.cfg["behaviors"] = behaviors
    return sp


def build_solvepack_from_config(config: str | Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None, seed: int | None = None) -> SolvePack:
    merged = config_io.load_and_merge(config, overrides or {})
    _fail_if_legacy(merged)
    params = config_io.extract_data_params(merged)
    rng = np.random.default_rng(seed)
    payload = _payload_from_params(params, rng)
    payload = _ensure_payload(payload, params, rng)
    steps_cfg = params.get("steps") or merged.get("timeline") or merged.get("behaviors")
    return _build_solvepack(payload, steps_cfg, merged.get("solver_cfg"), rng)


def load_solvepack_from_file(path: str, *, solver_cfg: dict | None = None, seed: int | None = None) -> SolvePack:
    rng = np.random.default_rng(seed)
    payload = file_io.load_scene(path)
    params = {
        "frame": {"width": payload.get("frame_size", (1, 1))[0], "height": payload.get("frame_size", (1, 1))[1]},
        "counts": {"labels": len(payload.get("labels", []))},
        "anchors": {"policy": "auto", "modes": {}},
    }
    payload = _ensure_payload(payload, params, rng)
    steps_cfg = payload.get("behaviors") or payload.get("steps")
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
    steps_cfg = params.get("steps")
    return _build_solvepack(payload, steps_cfg, solver_cfg, rng)
