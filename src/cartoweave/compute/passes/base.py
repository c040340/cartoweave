# -*- coding: utf-8 -*-
"""Base classes and data containers for compute passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Context:
    """Mutable execution context shared across passes."""

    pack: Any
    eval_index: int = 0
    step_index: int = 0
    bag: Dict[str, Any] = field(default_factory=dict)  # for pass communication


class ComputePass:
    """Pass base class.

    Subclasses may override hooks to wrap the energy function, wrap the
    step function or execute per-step logic via ``run``. All hooks are
    optional and default to pass-through behaviour.
    """

    name: str = ""

    def run(self, ctx: Context):  # optional
        return None

    def wrap_energy(self, energy_fn):  # optional
        return energy_fn

    def wrap_step(self, step_fn):  # optional
        return step_fn

    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool:  # optional
        return False
