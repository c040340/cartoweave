from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple
import numpy as np

EPS = 10.0 ** -12

# Signature of the energy/grad function for the CURRENT behavior state
EnergyFn = Callable[[np.ndarray, Any, Any, np.ndarray, Dict[str, Any]],
                    Tuple[float, np.ndarray, Dict[str, Any]]]

@dataclass
class LoopContext:
    labels: Any        # current LabelState list
    scene: Any         # current Scene
    active: np.ndarray # (N,) bool
    cfg: Dict[str, Any]
    iters: int         # iterations for this behavior
    mode: str = "lbfgs"
    params: Dict[str, Any] = None

@dataclass
class StepReport:
    k: int
    it: int
    E: float
    g_inf: float
    x_inf: float


def clip_by_inf(arr: np.ndarray, max_norm: float | None) -> np.ndarray:
    if not max_norm or max_norm <= 0:
        return arr
    a = float(np.abs(arr).max())
    if a <= max_norm:
        return arr
    return arr * (max_norm / a)

def run_iters(P0: np.ndarray, ctx: LoopContext, energy_fn: EnergyFn,
              report: bool = False) -> Tuple[np.ndarray, List[StepReport]]:
    """Run a minimal gradient loop with basic stopping criteria.

    - Calls energy_fn(P, ctx.labels, ctx.scene, ctx.active, ctx.cfg)
    - Performs a simple gradient descent step
    - Stops on tolerance or max iterations.
    Returns (P_final, reports).
    """
    P = P0.astype(float, copy=True)
    reps: List[StepReport] = []
    step = float(ctx.params.get("step", 1e-2) if ctx.params else 1e-2)
    opt_cfg = {}
    if ctx.cfg:
        opt_cfg = ctx.cfg.get("solver", {}) if isinstance(ctx.cfg, dict) else getattr(ctx.cfg, "solver", {})
        if opt_cfg is None:
            opt_cfg = {}
    gtol = float(opt_cfg.get("gtol", 1e-6))
    ftol = float(opt_cfg.get("ftol", 10.0 ** -9))
    xtol = float(opt_cfg.get("xtol", 10.0 ** -9))
    if ctx.params and ctx.params.get("max_step_norm") is not None:
        max_step_norm = ctx.params.get("max_step_norm")
    else:
        max_step_norm = opt_cfg.get("max_step_norm")

    E_prev: float | None = None
    for it in range(ctx.iters):
        for _ in range(2):
            E, g, comps = energy_fn(P, ctx.labels, ctx.scene, ctx.active, ctx.cfg)
            if not (np.isnan(E) or np.isnan(g).any()):
                break
            step *= 0.5
        else:
            break

        g_inf = float(np.max(np.abs(g))) if g.size else 0.0
        x_inf = float(np.max(np.abs(P))) if P.size else 0.0
        if report:
            reps.append(StepReport(k=-1, it=it, E=E, g_inf=g_inf, x_inf=x_inf))

        if g_inf <= gtol:
            break
        if E_prev is not None and abs(E - E_prev) <= ftol:
            break
        if step * g_inf <= xtol:
            break

        if g.shape == P.shape:
            delta = -step * g
            delta = clip_by_inf(delta, max_step_norm)
            P = P + delta

        E_prev = E
    return P, reps
