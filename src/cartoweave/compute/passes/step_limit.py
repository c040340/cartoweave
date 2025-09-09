# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cartoweave.utils.logging import logger
from . import get_pass_cfg
from .base import ComputePass


class StepLimitPass(ComputePass):
    name = "step_limit"

    def __init__(self, max_step_norm: float | None = 1.5):
        self.max_step_norm = max_step_norm
        self.stats = {"clamped_steps": 0, "max_observed_step_norm": 0.0}

    def wrap_step(self, step_fn):
        pm = getattr(self, "pm", None)
        cfg = getattr(pm, "cfg", {}) if pm else {}
        conf = get_pass_cfg(cfg, "step_limit", {"max_step_norm": self.max_step_norm})
        maxn = conf.get("max_step_norm", self.max_step_norm)
        if not maxn or maxn <= 0:
            return step_fn

        maxn = float(maxn)
        logger.info("[step_limit] max_step_norm=%s", maxn)
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
                if pm is not None and hasattr(pm, "emit_event"):
                    pm.emit_event(
                        {
                            "pass": "step_limit",
                            "info": "clamped",
                            "norm": float(n),
                            "max_norm": float(maxn),
                            "scale": float(scale),
                            "global_iter": getattr(pm, "eval_index", 0),
                        }
                        )
                logger.debug(
                    "[step_limit] norm=%g max=%g scale=%g", n, maxn, scale
                )
                return P_new
            return P_prop

        return wrapped
