# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from .base import ComputePass, Context, Stage


class StepLimitPass(ComputePass):
    """Inject ``step_limit`` and ``damping`` into each stage's params."""

    def __init__(self, max_step: float | None = None, damping: float | None = None):
        self.max_step = max_step
        self.damping = damping
        self.stats: Dict[str, Any] = {
            "applied_stages": 0,
            "max_step": max_step,
            "damping": damping,
        }

    def mutate_stage(self, stage: Stage):
        """Modify ``stage.params`` in-place and record how many stages changed."""
        changed = False
        if self.max_step is not None:
            stage.params = dict(stage.params)
            stage.params["step_limit"] = float(self.max_step)
            changed = True
        if self.damping is not None:
            stage.params = dict(stage.params)
            stage.params["damping"] = float(self.damping)
            changed = True
        if changed:
            self.stats["applied_stages"] += 1
