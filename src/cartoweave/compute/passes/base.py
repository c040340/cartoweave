# -*- coding: utf-8 -*-
"""Base classes and data containers for compute passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Stage:
    """Configuration for a single optimization stage."""

    index: int
    params: Dict[str, Any]
    mask: np.ndarray  # shape == (L,)
    iters: Optional[int] = None  # optional ``max_iter`` override for engine solver
    solver: Optional[str] = None  # solver name per stage ("lbfgs"/"semi")


@dataclass
class Context:
    """Mutable execution context shared across passes."""

    pack: Any
    stages: List[Stage]
    eval_index: int = 0
    stage_index: int = 0
    bag: Dict[str, Any] = field(default_factory=dict)  # for pass communication


class ComputePass:
    """Pass base class.

    Subclasses may override hooks to plan stages, decide whether to capture a
    frame, or wrap the energy function.
    """

    def plan_stages(self, ctx: Context) -> List[Stage]: ...

    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool: ...

    def wrap_energy(self, energy_fn):
        """Return a wrapped energy function or the original ``energy_fn``."""
        return energy_fn
