# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_label_label_disk import term_ll_disk as _impl

@register("ll.disk")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    return _impl(scene, P, cfg, phase=phase)
