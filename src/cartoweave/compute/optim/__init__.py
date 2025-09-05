# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import numpy as np

from .lbfgs import run_lbfgs  # present after Step 3
try:
    from .semi_newton import run_semi_newton
except Exception:  # optional solver
    def run_semi_newton(*args, **kwargs):
        raise NotImplementedError("semi_newton solver not available")

_SOLVERS: Dict[str, Callable[..., Any]] = {
    "lbfgs": run_lbfgs,
    "semi_newton": run_semi_newton,
}

def run_solver(
    name: str,
    P0: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    params: Dict[str, Any],
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Dispatch to a compute-local solver by name."""
    fn = _SOLVERS.get(name)
    if fn is None:
        raise KeyError(f"Unknown solver '{name}'. Available: {sorted(_SOLVERS)}")
    return fn(P0, energy_fn, grad_fn, params, callback=callback)

__all__ = ["run_solver", "run_lbfgs", "run_semi_newton"]
