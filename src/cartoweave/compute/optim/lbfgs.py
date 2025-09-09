from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import numpy as np
from scipy.optimize import minimize

from ...utils.logging import logger
from ...utils.config import get as cfg_get


def build_lbfgs_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract L-BFGS options from compute config."""
    pgtol = float(cfg_get(cfg, "compute.solver.tuning.lbfgsb.lbfgs_pgtol", 1e-9))
    maxiter = int(cfg_get(cfg, "compute.solver.tuning.lbfgsb.lbfgs_maxiter", 400))
    return {"gtol": pgtol, "maxiter": maxiter}


def run_lbfgs(
    P0: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    params: Dict[str, Any],
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Lightweight L-BFGS solver operating on compute energies.

    Parameters
    ----------
    P0 : np.ndarray
        Initial positions ``(L,2)``.
    energy_fn, grad_fn : callable
        Callbacks returning energy ``float`` and gradient ``(L,2)`` respectively.
    params : dict
        Solver parameters, supporting ``lbfgs_pgtol`` and ``lbfgs_maxiter``.
    callback : callable, optional
        Invoked after each evaluation with ``{"P", "G", "E"}``.
    """
    P0 = np.asarray(P0, float)
    if P0.ndim != 2 or P0.shape[1] != 2:
        raise ValueError(f"P0 must be (L,2), got {P0.shape}")
    L = P0.shape[0]
    x0 = P0.reshape(-1).astype(float)

    pgtol = float(params.get("lbfgs_pgtol", 1e-9))
    maxiter = int(params.get("lbfgs_maxiter", 400))

    last_E = float(energy_fn(P0))
    last_G = np.asarray(grad_fn(P0), float)
    if callback is not None:
        callback({"P": P0, "G": last_G, "E": last_E})

    def fun(x: np.ndarray):
        nonlocal last_E, last_G
        P = x.reshape(L, 2)
        E = float(energy_fn(P))
        G = np.asarray(grad_fn(P), float)
        last_E = E
        last_G = G
        if callback is not None:
            callback({"P": P, "G": G, "E": E})
        return E, G.reshape(-1)

    res = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "gtol": pgtol},
    )
    P_opt = res.x.reshape(L, 2)
    g_inf = float(np.linalg.norm(last_G, np.inf))
    logger.info("L-BFGS done nit=%s nfev=%s", res.nit, res.nfev)
    converged = bool(g_inf <= pgtol and res.success)
    msg = str(res.message)
    stop_reason = msg
    if not converged:
        msg_up = msg.upper()
        if "ABNORMAL_TERMINATION_IN_LNSRCH" in msg_up or g_inf <= pgtol:
            stop_reason = "zero_descent_rate"
    return {
        "P": P_opt,
        "E": float(last_E),
        "iters": int(res.nit),
        "converged": converged,
        "stop_reason": stop_reason,
    }
