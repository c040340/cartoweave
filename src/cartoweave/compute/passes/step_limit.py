# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from .base import ComputePass


class StepLimitPass(ComputePass):
    name = "StepLimitPass"

    def __init__(self, max_step_norm: float | None = 1.5):
        self.max_step_norm = max_step_norm
        self.stats = {"clamped_steps": 0, "max_observed_step_norm": 0.0}

    def wrap_step(self, step_fn):
        if not self.max_step_norm or self.max_step_norm <= 0:
            return step_fn

        maxn = float(self.max_step_norm)
        stats = self.stats

        def wrapped(P_old, P_prop, metrics):
            dP = P_prop - P_old
            n = float(np.linalg.norm(dP))
            if n > stats["max_observed_step_norm"]:
                stats["max_observed_step_norm"] = n
            if n > maxn and n > 0:
                scale = maxn / n
                P_new = P_old + dP * scale
                stats["clamped_steps"] += 1
                if isinstance(metrics, dict):
                    metrics["step_limit_clamped"] = True
                    metrics["step_limit_scale"] = float(scale)
                return P_new
            return P_prop

        return wrapped
