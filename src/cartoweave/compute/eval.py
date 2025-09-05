"""Thin wrappers around core evaluation utilities."""

# -*- coding: utf-8 -*-
import numpy as np
from .types import Array2

# 兼容你仓库里已存在的实现（函数名可能是 energy_and_grad_full 或 energy_and_grad_fullP）
try:
    from cartoweave.engine.core_eval import energy_and_grad_full as _core_energy
except Exception:
    try:
        from cartoweave.engine.core_eval import energy_and_grad_fullP as _core_energy
    except Exception:
        _core_energy = None  # 延迟到调用时报错

def energy_and_grad_full(P: Array2, scene, active_mask: np.ndarray, cfg: dict):
    """
    内部评价入口：薄包装，调用 engine/core_eval 的已实现版本。
    返回 (E_total, G_full, comps_full, meta)，且 G/comps 为全长 (L,2)。
    """
    if _core_energy is None:
        raise RuntimeError("Cannot import core energy function from cartoweave.engine.core_eval")
    return _core_energy(P, scene, active_mask, cfg)
