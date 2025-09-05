# -*- coding: utf-8 -*-
"""Wrapper for area embed force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_area_embed import term_area_embed as _impl


@register("area.embed")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the area embed term."""

    return _impl(scene, P, cfg, phase=phase)
