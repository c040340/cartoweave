# -*- coding: utf-8 -*-
"""Pass registry and builder utilities."""

from __future__ import annotations
from typing import Any, Dict, List, Union
from .base import Context, Stage, ComputePass
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


def build_passes(cfg_list: List[Union[str, Dict]] | None, default_capture_cfg: Dict) -> List[ComputePass]:
    """Instantiate passes from configuration.

    ``cfg_list`` may contain strings or ``{"name": ..., "args": ...}``
    dictionaries. ``CapturePass`` is added with ``default_capture_cfg`` if not
    specified. ``SchedulePass`` is always present (single stage by default).
    """
    passes: List[ComputePass] = []
    names = set()

    def _mk(name: str, args: Dict | None):
        cls = REGISTRY.get(name)
        if not cls:
            raise ValueError(f"Unknown pass: {name}")
        needs_args = name in ("capture", "weights", "grad_clip", "nan_guard", "step_limit")
        inst = cls(**(args or {})) if needs_args else cls()
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
        passes.append(CapturePass(**default_capture_cfg))
    return passes
