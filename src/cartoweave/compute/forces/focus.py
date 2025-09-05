# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_focus import term_focus_huber as _impl

@register("focus.attract")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    return _impl(scene, P, cfg, phase=phase)
