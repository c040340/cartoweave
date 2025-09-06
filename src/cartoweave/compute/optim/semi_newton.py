from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import numpy as np

from ...utils.logging import logger

Array = np.ndarray


def _cfg(cfg: Dict[str, Any], key: str, default):
    return cfg[key] if key in cfg else default


def _hvp_fd(grad_fn: Callable[[Array], Array], x: Array, v: Array, eps: float) -> Array:
    tiny = np.finfo(float).eps
    return (grad_fn(x + eps * v) - grad_fn(x)) / max(eps, tiny)


def _cg_solve(apply_A: Callable[[Array], Array], b: Array, tol: float, maxit: int) -> tuple[Array, int]:
    x = np.zeros_like(b)
    r = b - apply_A(x)
    p = r.copy()
    rr = float(np.dot(r, r))
    if rr == 0.0:
        return x, 0
    tiny = np.finfo(float).tiny
    b_norm2 = float(np.dot(b, b)) + tiny
    for k in range(maxit):
        Ap = apply_A(p)
        denom = float(np.dot(p, Ap)) + tiny
        if denom <= 0.0 or not np.isfinite(denom):
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(np.dot(r, r))
        if rr_new <= (tol * tol) * b_norm2:
            return x, k + 1
        beta = rr_new / (rr + tiny)
        p = r + beta * p
        rr = rr_new
    return x, maxit


def run_semi_newton(
    P0: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    params: Dict[str, Any],
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Finite-difference semi-Newton optimiser.

    Parameters are similar to engine version but operate directly on compute
    callbacks. Only the basic algorithm is implemented to satisfy tests.
    """
    P = np.asarray(P0, float).copy()
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"P0 must be (L,2), got {P.shape}")
    L = P.shape[0]

    max_outer = int(_cfg(params, "sn_max_outer", 60))
    dt = float(_cfg(params, "sn_dt", 1.0))
    hvp_eps = float(_cfg(params, "sn_hvp_eps", 1e-4))
    cg_tol = float(_cfg(params, "sn_cg_tol", 1e-4))
    cg_maxit = int(_cfg(params, "sn_cg_maxit", 40))
    lm = float(_cfg(params, "sn_lm0", 1e-3))
    gtol = float(_cfg(params, "sn_gtol", 1e-3))
    armijo_c1 = float(_cfg(params, "sn_armijo_c1", 1e-4))
    max_backtrack = int(_cfg(params, "sn_max_backtrack", 8))

    x = P.reshape(-1)

    def _rec(P_full: Array, E: float, G_full: Array):
        if callback is not None:
            callback({"P": P_full, "G": G_full, "E": float(E)})

    def fun_full(P_full: Array) -> tuple[float, Array]:
        E = float(energy_fn(P_full))
        G = np.asarray(grad_fn(P_full), float)
        _rec(P_full, E, G)
        return E, G

    def fun_vars(x_vars: Array) -> tuple[float, Array]:
        P_full = x_vars.reshape(L, 2)
        E, G = fun_full(P_full)
        return E, G.reshape(-1)

    for it in range(max_outer):
        E, g = fun_vars(x)
        g_inf = float(np.linalg.norm(g, np.inf))
        logger.debug("Semi-Newton iter %d E=%.6g g_inf=%.6g", it, float(E), g_inf)
        if g_inf <= gtol:
            logger.info("Semi-Newton converged nit=%d g_inf=%.6g", it, g_inf)
            return {
                "P": x.reshape(L, 2),
                "E": float(E),
                "iters": it,
                "converged": True,
                "stop_reason": "gtol",
            }

        grad_only = lambda z: fun_vars(z)[1]

        def apply_A(vec: Array) -> Array:
            Hv = _hvp_fd(lambda z: grad_only(z), x, vec, hvp_eps)
            return vec + dt * Hv + lm * vec

        rhs = -dt * g
        d, _ = _cg_solve(apply_A, rhs, tol=cg_tol, maxit=cg_maxit)

        alpha = 1.0
        gTd = float(np.dot(g, d))
        accepted = False
        for _ in range(max_backtrack + 1):
            x_try = x + alpha * d
            E_try, _ = fun_vars(x_try)
            if E_try <= E + armijo_c1 * alpha * gTd:
                x = x_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            dt *= 0.5
            if dt < 1e-3:
                break

    E_final, g_final = fun_vars(x)
    g_inf_final = float(np.linalg.norm(g_final, np.inf))
    logger.info("Semi-Newton done nit=%d g_inf=%.6g", max_outer, g_inf_final)
    return {
        "P": x.reshape(L, 2),
        "E": float(E_final),
        "iters": max_outer,
        "converged": bool(g_inf_final <= gtol),
        "stop_reason": "max_iter",
    }
