from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from cartoweave.utils.dict_merge import deep_update

from .config import AreaGenCfg, RouteGenCfg

__all__ = ["load_and_merge", "extract_data_params"]


def load_and_merge(config: str | Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict:
    """Load a YAML config or copy a dict and deep-merge overrides."""
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf8") as fh:
            base = yaml.safe_load(fh) or {}
    elif isinstance(config, Mapping):
        base = dict(config)
    else:
        raise TypeError("config must be path or mapping")
    if overrides:
        base = deep_update(base, overrides)
    return base


def _route_defaults() -> dict:
    cfg = RouteGenCfg()
    return {
        "segment_len_scale": float(cfg.segment_len_scale),
        "inset_margin_scale": float(cfg.inset_margin_scale),
        "min_vertex_spacing_scale": float(cfg.min_vertex_spacing_scale),
    }


def _area_defaults() -> dict:
    cfg = AreaGenCfg()
    return {
        "inset_margin_scale": float(cfg.inset_margin_scale),
        "min_edge_spacing_scale": float(cfg.min_edge_spacing_scale),
    }


def extract_data_params(cfg: Mapping[str, Any]) -> dict:
    """Extract and normalize the ``data`` section from *cfg*.

    Legacy ``timeline`` is mapped to ``steps`` with a warning.
    """
    data = dict(cfg.get("data", {}))
    if "timeline" in data and "steps" not in data:
        data["steps"] = data.pop("timeline")
        warnings.warn("Use 'steps' instead of 'timeline'", UserWarning, stacklevel=2)

    frame = data.get("frame", {})
    counts = data.get("counts", {})
    rand = data.get("random", {})
    anchors = data.get("anchors", {})
    steps_cfg = data.get("steps", {})

    route_defaults = _route_defaults()
    area_defaults = _area_defaults()

    route_gen = {k: float(rand.get("route_gen", {}).get(k, route_defaults[k])) for k in route_defaults}
    area_gen = {k: float(rand.get("area_gen", {}).get(k, area_defaults[k])) for k in area_defaults}

    modes_in = anchors.get("modes", {})
    modes = {
        "line": modes_in.get("line", "projected"),
        "area": modes_in.get("area", "projected_edge"),
    }

    steps_norm = {
        "kind": steps_cfg.get("kind", "none"),
        "steps": steps_cfg.get("steps"),
    }
    if "group_sizes" in steps_cfg:
        steps_norm["group_sizes"] = list(steps_cfg["group_sizes"])
    if "groups" in steps_cfg:
        steps_norm["groups"] = [list(g) for g in steps_cfg["groups"]]

    return {
        "source": data.get("source", "random"),
        "path": data.get("path", ""),
        "regen": bool(data.get("regen", False)),
        "frame": {
            "width": int(frame.get("width", 800)),
            "height": int(frame.get("height", 600)),
        },
        "counts": {
            "labels": int(counts.get("labels", 0)),
            "points": int(counts.get("points", 0)),
            "lines": int(counts.get("lines", 0)),
            "areas": int(counts.get("areas", 0)),
        },
        "random": {
            "route_gen": route_gen,
            "area_gen": area_gen,
        },
        "anchors": {
            "policy": anchors.get("policy", "auto"),
            "modes": modes,
        },
        "steps": steps_norm,
    }
