# -*- coding: utf-8 -*-
"""Pass registry and builder utilities."""

from __future__ import annotations
from typing import Any, Dict, List, Union
from .base import Context, Stage, ComputePass


def get_pass_cfg(cfg: dict, name: str, defaults: dict | None = None) -> dict:
    d = (cfg.get("compute", {}).get("passes", {}).get(name) or {})
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

REGISTRY = {
    "schedule": SchedulePass,
    "capture":  CapturePass,
    "weights":  WeightsPass,
    "nan_guard": NaNGuardPass,
    "grad_clip": GradClipPass,
    "step_limit": StepLimitPass,
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
        cls = REGISTRY.get(name)
        if not cls:
            raise ValueError(f"Unknown pass: {name}")
        if name == "weights":
            inst = cls(args or {})
        elif name == "step_limit":
            inst = cls(cfg)
        elif args:
            inst = cls(**args)
        else:
            inst = cls()
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
        passes.insert(0, SchedulePass())
        names.add("schedule")
    if "capture" not in names:
        passes.append(CapturePass())
    return passes
