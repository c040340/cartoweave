# -*- coding: utf-8 -*-
"""Wrapper for label–label disk force."""

from __future__ import annotations
import numpy as np
from . import register
from cartoweave.engine.forces.term_label_label_disk import term_ll_disk as _impl


@register("ll.disk")
def evaluate(scene, P: np.ndarray, cfg: dict, phase: str):
    """Return energy, force field and meta for the label–label disk term."""

    return _impl(scene, P, cfg, phase=phase)
