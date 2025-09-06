# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional, Tuple
import numpy as np

from . import build_passes


class PassManager:
    """
    Central coordinator for compute passes.
    Responsibilities:
      - instantiate passes from spec (via build_passes)
      - let passes plan/mutate stages
      - wrap energy/grad chain
      - wrap step function (optional)
      - coordinate capture decisions and collect stats
    """
    def __init__(self, cfg: dict, passes_spec: List[Any]):
        self.cfg = cfg
        self.passes = build_passes(cfg, passes_spec)

    # ---------- stages ----------
    def plan_stages(self, ctx: Any, base_stages: List[Any]) -> List[Any]:
        stages = list(base_stages or [])
        # let passes create/adjust stages first (plan)
        for p in self.passes:
            plan = getattr(p, "plan_stages", None)
            if callable(plan):
                stages = plan(ctx, stages)
        # then give each pass a chance to mutate each stage
        mutate = [(p, getattr(p, "mutate_stage", None)) for p in self.passes]
        out: List[Any] = []
        for st in stages:
            st2 = st
            for p, fn in mutate:
                if callable(fn):
                    st2 = fn(ctx, st2)
            out.append(st2)
        return out

    # ---------- wrapping ----------
    def wrap_energy(
        self,
        energy_fn: Callable[[np.ndarray, Any, Any, np.ndarray, Dict[str, Any]], Tuple[float, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]],
    ) -> Callable[[np.ndarray, Any, Any, np.ndarray, Dict[str, Any]], Tuple[float, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]]:
        wrapped = energy_fn
        for p in self.passes:
            fn = getattr(p, "wrap_energy", None)
            if callable(fn):
                wrapped = fn(wrapped)
        return wrapped

    def wrap_step(self, step_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]) \
            -> Callable[[np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]:
        """
        step_fn: (P_old, P_proposed, iter_metrics) -> P_applied
        Default pipeline is identity; passes can clamp/modify step (e.g., StepLimitPass).
        """
        wrapped = step_fn
        for p in self.passes:
            fn = getattr(p, "wrap_step", None)
            if callable(fn):
                wrapped = fn(wrapped)
        return wrapped

    # ---------- capture ----------
    def want_capture(self, ctx: Any, eval_i: int, frames_len: int) -> bool:
        for p in self.passes:
            fn = getattr(p, "want_capture", None)
            if callable(fn) and fn(ctx, eval_i, frames_len):
                return True
        return False

    # ---------- stats ----------
    def collect_stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in self.passes:
            name = getattr(p, "name", "") or p.__class__.__name__
            stats = getattr(p, "stats", None)
            if stats:
                out[name] = stats
        return out
