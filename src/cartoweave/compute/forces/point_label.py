# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_point_label import term_point_label as _impl

@register("pl.rect")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    return _impl(scene, P, cfg, phase=phase)
