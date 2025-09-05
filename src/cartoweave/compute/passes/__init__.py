# -*- coding: utf-8 -*-
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
    """
    cfg_list 允许：
      - "schedule"
      - {"name":"schedule","args":{...}}
      - "capture"
      - {"name":"capture","args":{"every":5,"limit":100}}
    未提供 capture 时，会用 default_capture_cfg 补一个 CapturePass。
    schedule 若未提供，则默认也会加上（单阶段）。
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
