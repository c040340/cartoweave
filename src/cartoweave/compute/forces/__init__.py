# -*- coding: utf-8 -*-
"""Compute-side force registry and wrappers.

Each force term lives in ``compute/forces/<name>.py`` and registers an
``evaluate`` function returning ``(E, F, meta)`` where ``F`` is a force field of
shape ``(L, 2)``. The compute aggregator converts these forces into gradients.
"""

from __future__ import annotations
from typing import Dict, Callable

REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Decorator registering a compute-side force implementation."""

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
