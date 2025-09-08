import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import run_solver

EPS = 10.0 ** -12

OnIter = Optional[Callable[[int, np.ndarray, Dict[str, Any]], None]]

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

def run_iters(
    P0: np.ndarray,
    ctx: LoopContext,
    energy_fn: EnergyFn,
    report: bool = False,
    on_iter: OnIter = None,
) -> Tuple[np.ndarray, List[StepReport]]:
    """Run optimisation iterations depending on ``ctx.mode``.

    For ``lbfgs``/``lbfgsb`` and ``semi_newton`` modes the dedicated solver is
    dispatched via :func:`run_solver`.  Otherwise a minimal gradient-descent loop
    is executed, matching the previous behaviour.
    """

    comp = (ctx.cfg or {}).get("compute", {}) if ctx.cfg else {}

    if ctx.mode in {"lbfgs", "lbfgsb", "semi_newton"}:
        # ------------------------------------------------------------------
        # Delegation to specialised solvers
        # ------------------------------------------------------------------
        solver_name = "lbfgs" if ctx.mode in {"lbfgs", "lbfgsb"} else ctx.mode
        last: Dict[str, Any] = {}

        def _eval(P: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, Any]]:
            E, G, comps = energy_fn(P, ctx.labels, ctx.scene, ctx.active, comp)
            last.update({"E": E, "G": G, "P": P, "comps": comps})
            return E, G, comps

        def _energy(P: np.ndarray) -> float:
            E, _, _ = _eval(P)
            return float(E)

        def _grad(P: np.ndarray) -> np.ndarray:
            _, G, _ = _eval(P)
            return np.asarray(G, float)

        reps: List[StepReport] = []
        k = 0

        def _cb(info: Dict[str, Any]):
            nonlocal k
            g = np.asarray(info.get("G"), float)
            P_iter = np.asarray(info.get("P"), float)
            E_iter = float(info.get("E", 0.0))
            g_inf = float(np.max(np.abs(g))) if g.size else 0.0
            x_inf = float(np.max(np.abs(P_iter))) if P_iter.size else 0.0
            if report:
                reps.append(StepReport(k=-1, it=k, E=E_iter, g_inf=g_inf, x_inf=x_inf))
            if on_iter is not None:
                comps_iter = last.get("comps", {})
                try:
                    on_iter(k, P_iter, {"E": E_iter, "G": g, "comps": comps_iter})
                except Exception:
                    pass
            k += 1

        res = run_solver(solver_name, P0, _energy, _grad, ctx.params or {}, callback=_cb)
        P_final = np.asarray(res.get("P", P0), float)
        if report and not reps:
            # ensure at least one report for downstream consumers
            E_fin, G_fin, _ = _eval(P_final)
            g_inf_fin = float(np.max(np.abs(G_fin))) if G_fin.size else 0.0
            x_inf_fin = float(np.max(np.abs(P_final))) if P_final.size else 0.0
            reps.append(
                StepReport(
                    k=-1,
                    it=int(res.get("iters", 0)),
                    E=float(E_fin),
                    g_inf=g_inf_fin,
                    x_inf=x_inf_fin,
                )
            )
        return P_final, reps

    # ----------------------------------------------------------------------
    # Basic gradient descent fallback
    # ----------------------------------------------------------------------
    P = P0.astype(float, copy=True)
    reps: List[StepReport] = []
    step = float(ctx.params.get("step", 1e-2) if ctx.params else 1e-2)
    solver_cfg = comp.get("solver") or {}
    stop_cfg = ((solver_cfg.get("tuning") or {}).get("stop") or {})
    gtol = float(stop_cfg.get("gtol", 1.0e-4))
    ftol = float(stop_cfg.get("ftol", 1.0e-9))
    xtol = float(stop_cfg.get("xtol", 1.0e-9))
    max_stall = stop_cfg.get("max_stall_iters")
    if ctx.params and ctx.params.get("max_step_norm") is not None:
        max_step_norm = ctx.params.get("max_step_norm")
    else:
        max_step_norm = solver_cfg.get("max_step_norm") if solver_cfg else None

    E_prev: float | None = None
    stall_iters = 0
    for it in range(ctx.iters):
        for _ in range(2):
            E, g, comps = energy_fn(P, ctx.labels, ctx.scene, ctx.active, comp)
            if not (np.isnan(E) or np.isnan(g).any()):
                break
            step *= 0.5
        else:
            break

        g_inf = float(np.max(np.abs(g))) if g.size else 0.0
        x_inf = float(np.max(np.abs(P))) if P.size else 0.0
        if report:
            reps.append(StepReport(k=-1, it=it, E=E, g_inf=g_inf, x_inf=x_inf))

        if on_iter is not None:
            try:
                on_iter(it, P, {"E": E, "G": g, "comps": comps})
            except Exception:
                pass

        if g_inf <= gtol:
            break
        dE = abs(E - E_prev) if E_prev is not None else None
        if dE is not None and dE <= ftol:
            stall_iters += 1
            if max_stall is None or stall_iters >= int(max_stall):
                break
        else:
            stall_iters = 0
        if step * g_inf <= xtol:
            break

        if g.shape == P.shape:
            delta = -step * g
            delta = clip_by_inf(delta, max_step_norm)
            P = P + delta

        E_prev = E
    return P, reps
