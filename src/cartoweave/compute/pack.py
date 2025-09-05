# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .types import Array2, EnergyFn


@dataclass
class SolvePack:
    # 维度与初值
    L: int
    P0: Array2
    active_mask0: np.ndarray  # shape == (L,)

    # 只读场景对象（你的现有类型），不在 compute 内修改
    scene: object

    # 求解模式与参数（透传给优化器/评价）
    mode: str = "lbfgs"
    params: Dict = field(default_factory=dict)

    # 评价回调（默认在 compute.eval 中提供）
    energy_and_grad: Optional[EnergyFn] = None

    # 阶段/增强器配置（后续步骤使用）
    schedule: Optional[List[Dict]] = None
    passes: Optional[List] = None

    # 记录采样策略（后续步骤使用）
    capture: Dict = field(default_factory=lambda: {"every": 1, "limit": None, "final_always": True})

    def validate(self) -> None:
        assert isinstance(self.L, int) and self.L > 0, "L must be positive int"
        assert isinstance(self.P0, np.ndarray) and self.P0.shape == (self.L, 2), "P0 shape must be (L,2)"
        assert isinstance(self.active_mask0, np.ndarray) and self.active_mask0.shape == (self.L,), "active_mask0 shape must be (L,)"
        assert self.P0.dtype.kind in "fc", "P0 must be float/complex (float expected)"
        # 非活跃梯度零的约束在评价/优化器层保证；此处只校形状
