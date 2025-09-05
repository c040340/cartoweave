# -*- coding: utf-8 -*-
"""Wrapper for point-label force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_point_label import term_point_label as _impl


@register("pl.rect")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the point-label term."""

    return _impl(scene, P, cfg, phase=phase)
