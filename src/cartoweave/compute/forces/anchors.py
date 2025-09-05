# -*- coding: utf-8 -*-
"""Wrapper for anchor spring force."""

from __future__ import annotations
import numpy as np
from . import register

from cartoweave.engine.forces.term_anchor import term_anchor as _impl


@register("anchor.spring")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the anchor spring term."""

    return _impl(scene, P, cfg, phase=phase)
