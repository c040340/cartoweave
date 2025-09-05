# -*- coding: utf-8 -*-
"""Wrapper for area soft-out force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_area_softout import term_area_softout as _impl


@register("area.softout")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the area soft-out term."""

    return _impl(scene, P, cfg, phase=phase)
