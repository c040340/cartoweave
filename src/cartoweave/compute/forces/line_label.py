# -*- coding: utf-8 -*-
"""Wrapper for line-label force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_line_label import term_line_label as _impl


@register("ln.rect")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the line-label term."""

    return _impl(scene, P, cfg, phase=phase)
