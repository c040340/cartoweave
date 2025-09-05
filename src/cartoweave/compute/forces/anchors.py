# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register

# 直接复用旧引擎 term 实现（薄包装）
from cartoweave.engine.forces.term_anchor import term_anchor as _impl

@register("anchor.spring")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    # 旧实现签名：term_(scene, P, cfg, phase)
    # 返回: (E, F, source_meta)
    return _impl(scene, P, cfg, phase=phase)
