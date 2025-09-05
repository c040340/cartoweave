# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Stage:
    index: int
    params: Dict[str, Any]
    mask: np.ndarray  # shape == (L,)
    iters: Optional[int] = None  # 仅用于传给 legacy solver 的 max_iter 覆盖
    solver: Optional[str] = None   # 每阶段选择 solver（"lbfgs"/"semi"）


@dataclass
class Context:
    pack: Any
    stages: List[Stage]
    eval_index: int = 0
    stage_index: int = 0
    # 可在各 pass 间共享信息或记录统计
    bag: Dict[str, Any] = field(default_factory=dict)


class ComputePass:
    """Pass 基类，可选择性覆写以下钩子。"""
    def plan_stages(self, ctx: Context) -> List[Stage]: ...
    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool: ...

    # 能量包装钩子：默认直接返回原函数，子类可覆写
    def wrap_energy(self, energy_fn):
        return energy_fn
