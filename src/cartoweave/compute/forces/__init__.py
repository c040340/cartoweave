# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Callable

# 我们自己的（计算侧）注册表：名字 → evaluate(scene, P, cfg, phase) -> (E, F, meta)
# 注意：此处每个 evaluate 返回的是“力（force）场 F=(L,2)”，而非梯度；
# 聚合器会做 g -= F 与 comps[name] = F。
REGISTRY: Dict[str, Callable] = {}

def register(name: str):
    def deco(fn: Callable):
        REGISTRY[name] = fn
        return fn
    return deco

# 显式导入我们迁移/包装过的项
from .anchors import evaluate as _anchors_eval  # anchor.spring
from .boundary import evaluate as _boundary_eval  # boundary.wall
from .label_label_disk import evaluate as _lldisk_eval  # ll.disk
from .focus import evaluate as _focus_eval  # focus.attract
from .point_label import evaluate as _pl_eval  # pl.rect
from .line_label import evaluate as _lline_eval  # ln.rect
from .area_embed import evaluate as _aembed_eval  # area.embed
from .area_cross import evaluate as _across_eval  # area.cross
from .area_softout import evaluate as _asoft_eval  # area.softout
from .label_label_rect import evaluate as _llrect_eval  # ll.rect

# 暴露 REGISTRY 即可
__all__ = ["REGISTRY", "register"]
