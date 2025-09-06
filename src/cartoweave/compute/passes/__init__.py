# -*- coding: utf-8 -*-
"""Pass registry and builder utilities."""

from __future__ import annotations
from typing import Any, Dict, List, Union
from .base import Context, Stage, ComputePass


def get_pass_cfg(cfg: dict, name: str, defaults: dict | None = None) -> dict:
    d = (cfg.get("passes", {}).get(name) or {})
    if defaults:
        out = dict(defaults)
        out.update(d)
        return out
    return dict(d)

from .schedule import SchedulePass
from .capture import CapturePass
from .weights import WeightsPass
from .nan_guard import NaNGuardPass
from .grad_clip import GradClipPass
from .step_limit import StepLimitPass
from .geom_preproc import GeomPreprocPass

REGISTRY = {
    "schedule":   (SchedulePass, {}),
    "weights":    (WeightsPass, {}),
    "nan_guard":  (NaNGuardPass, {"on_nan": "zero", "on_inf": "clip"}),
    "grad_clip":  (GradClipPass, {"max_norm": None, "max_abs": None}),
    "step_limit": (StepLimitPass, {"max_step_norm": 1.5}),
    "capture":    (CapturePass, {"every": 1, "final_always": True}),
    "geom_preproc": (GeomPreprocPass, {"tiny_eps": 1e-9}),
}


def build_passes(cfg: Dict, cfg_list: List[Union[str, Dict]] | None) -> List[ComputePass]:
    """Instantiate passes from specification.

    ``cfg_list`` may contain strings or ``{"name": ..., "args": ...}``
    dictionaries. ``CapturePass`` is added with defaults from ``cfg`` if not
    specified. ``SchedulePass`` is always present (single stage by default).
    """
    passes: List[ComputePass] = []
    names = set()

    def _mk(name: str, args: Dict | None):
        entry = REGISTRY.get(name)
        if not entry:
            raise ValueError(f"Unknown pass: {name}")
        cls, defaults = entry
        kwargs = dict(defaults or {})
        if args:
            kwargs.update(args)
        inst = cls(**kwargs)
        passes.append(inst)
        names.add(name)

    cfg_list = cfg_list or []
    for item in cfg_list:
        if isinstance(item, str):
            _mk(item, None)
        elif isinstance(item, dict):
            _mk(item.get("name"), item.get("args"))
        else:
            raise ValueError("pass config must be str or dict")

    if "schedule" not in names:
        cls, defaults = REGISTRY["schedule"]
        passes.insert(0, cls(**defaults))
        names.add("schedule")
    if "geom_preproc" not in names:
        cls, defaults = REGISTRY["geom_preproc"]
        passes.insert(1, cls(**defaults))
        names.add("geom_preproc")
    if "step_limit" not in names:
        cls, defaults = REGISTRY["step_limit"]
        passes.append(cls(**defaults))
        names.add("step_limit")
    if "capture" not in names:
        cls, defaults = REGISTRY["capture"]
        passes.append(cls(**defaults))
    return passes
