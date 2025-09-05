# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import copy

from .base import ComputePass, Context


class WeightsPass(ComputePass):
    """Inject static per-term weights into the evaluation config."""

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {}
        self.stats: Dict[str, float] = {}

    def _weight_for(self, name: str) -> float:
        if name in self.weights:
            return float(self.weights[name])
        for k, v in self.weights.items():
            if k.endswith(".*") and name.startswith(k[:-2] + "."):
                return float(v)
            if "." not in k and name.startswith(k + "."):
                return float(v)
        return 1.0

    def wrap_energy(self, energy_fn):
        """Attach weight mapping to the config before evaluation."""

        def _wrapped(P, scene, active_mask, cfg):
            cfg2 = copy.deepcopy(cfg) if isinstance(cfg, dict) else {}
            cfg2.setdefault("compute", {}).setdefault("weights", {})
            cfg2["compute"]["weights"].update(self.weights)
            return energy_fn(P, scene, active_mask, cfg2)

        return _wrapped
