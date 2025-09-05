# -*- coding: utf-8 -*-
"""Wrapper for area cross force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_area_cross import term_area_cross as _impl


@register("area.cross")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the area cross term."""

    return _impl(scene, P, cfg, phase=phase)
