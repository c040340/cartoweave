from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np

from ..core_eval import energy_and_grad_fullP
from ...utils.logging import logger

Array = np.ndarray

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def _cfg(cfg: Dict[str, Any], key: str, default):
    """Read a value from cfg with a fallback without triggering StrictConfig."""
    return cfg[key] if key in cfg else default


def _hvp_fd(grad_fn: Callable[[Array], Array], x: Array, v: Array, eps: float) -> Array:
    """Finite-difference Hessian-vector product: H v ≈ (g(x+eps v) - g(x)) / eps"""
    return (grad_fn(x + eps * v) - grad_fn(x)) / max(eps, 1e-12)


def _cg_solve(apply_A: Callable[[Array], Array], b: Array, tol: float, maxit: int) -> Tuple[Array, int]:
    """Conjugate gradient solve for A x = b."""
    x = np.zeros_like(b)
    r = b - apply_A(x)
    p = r.copy()
    rr = float(np.dot(r, r))
    if rr == 0.0:
        return x, 0
    b_norm2 = float(np.dot(b, b)) + 1e-18
    for k in range(maxit):
        Ap = apply_A(p)
        denom = float(np.dot(p, Ap)) + 1e-18
        if denom <= 0.0 or not np.isfinite(denom):
            break
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(np.dot(r, r))
        if rr_new <= (tol * tol) * b_norm2:
            return x, k + 1
        beta = rr_new / (rr + 1e-18)
        p = r + beta * p
        rr = rr_new
    return x, maxit

# ----------------------------------------------------------------------------
# Semi-Newton solver
# ----------------------------------------------------------------------------

def solve_layout_semi_newton(scene, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """A lightweight semi-Newton solver using finite-difference Hessian probes.

    The algorithm optimises only the indices listed in ``scene['movable_idx']``.
    Other labels remain fixed.  Behaviour is intentionally minimal yet robust
    enough for unit tests and small scenes.
    """
    labels_init = scene.get("labels_init")
    N = 0 if labels_init is None else int(labels_init.shape[0])
    if N == 0:
        logger.info("Semi-Newton early exit: no labels")
        return np.zeros((0, 2), float), {
            "nit": 0,
            "success": True,
            "msg": "no labels",
            "history": {"positions": [], "energies": [], "records": []},
        }

    mov = scene.get("movable_idx")
    if mov is None:
        mov = np.arange(N, dtype=int)
    else:
        mov = np.asarray(mov, dtype=int)

    # Algorithm parameters ---------------------------------------------------
    max_outer = int(_cfg(cfg, "sn_max_outer", 60))
    dt = float(_cfg(cfg, "sn_dt", 1.0))
    hvp_eps = float(_cfg(cfg, "sn_hvp_eps", 1e-4))
    cg_tol = float(_cfg(cfg, "sn_cg_tol", 1e-4))
    cg_maxit = int(_cfg(cfg, "sn_cg_maxit", 40))
    lm = float(_cfg(cfg, "sn_lm0", 1e-3))
    gtol = float(_cfg(cfg, "sn_gtol", 1e-3))
    armijo_c1 = float(_cfg(cfg, "sn_armijo_c1", 1e-4))
    max_backtrack = int(_cfg(cfg, "sn_max_backtrack", 8))

    P = np.asarray(labels_init, float).copy()
    x = P[mov].reshape(-1)

    history = {"positions": [], "energies": [], "records": []}
    def _recorder(P, E, comps, meta):
        history["records"].append({
            "P": P.copy(),
            "E": float(E),
            "comps": {k: v.copy() for k, v in comps.items()},
            "meta": dict(meta) if meta else {},
        })
    logger.info("Semi-Newton start n_labels=%d max_outer=%d", N, max_outer)

    def fun_full(P_full: Array, *, record_cb=_recorder) -> Tuple[float, Array]:
        E, G, _ = energy_and_grad_fullP(scene, P_full, cfg, record=record_cb)
        return E, G

    def fun_vars(x_vars: Array, *, record_cb=_recorder) -> Tuple[float, Array]:
        P_full = P.copy()
        P_full[mov] = x_vars.reshape(-1, 2)
        E, G = fun_full(P_full, record_cb=record_cb)
        return E, G[mov].reshape(-1)

    for it in range(max_outer):
        E, g_mov = fun_vars(x)
        history["positions"].append(P.copy())
        history["energies"].append(float(E))
        g_inf = float(np.linalg.norm(g_mov, np.inf))
        meta = history["records"][-1].setdefault("meta", {})
        meta["solver_info"] = {
            "solver": "semi",
            "g_inf": g_inf,
            "gtol": gtol,
            "iter": it,
            "iter_max": max_outer,
        }
        logger.debug("Semi-Newton iter %d E=%.6g g_inf=%.6g", it, float(E), g_inf)
        if g_inf <= gtol:
            info = {"nit": it, "success": True, "g_inf": g_inf, "history": history}
            logger.info("Semi-Newton converged nit=%d g_inf=%.6g", it, g_inf)
            return P, info

        grad_only = lambda z: fun_vars(z, record_cb=None)[1]

        def apply_A(vec: Array) -> Array:
            Hv = _hvp_fd(grad_only, x, vec, hvp_eps)
            return vec + dt * Hv + lm * vec

        rhs = -dt * g_mov
        d, _ = _cg_solve(apply_A, rhs, tol=cg_tol, maxit=cg_maxit)

        alpha = 1.0
        gTd = float(np.dot(g_mov, d))
        accepted = False
        for _ in range(max_backtrack + 1):
            x_try = x + alpha * d
            P_try = P.copy()
            P_try[mov] = x_try.reshape(-1, 2)
            E_try, _ = fun_full(P_try, record_cb=None)
            if E_try <= E + armijo_c1 * alpha * gTd:
                x = x_try
                P = P_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            dt *= 0.5
            if dt < 1e-3:
                break

    # final evaluation
    E_final, g_mov_final = fun_vars(x)
    history["positions"].append(P.copy())
    history["energies"].append(float(E_final))
    g_inf_final = float(np.linalg.norm(g_mov_final, np.inf))
    meta = history["records"][-1].setdefault("meta", {})
    meta["solver_info"] = {
        "solver": "semi",
        "g_inf": g_inf_final,
        "gtol": gtol,
        "iter": max_outer,
        "iter_max": max_outer,
    }
    info = {
        "nit": max_outer,
        "success": g_inf_final <= gtol,
        "g_inf": g_inf_final,
        "message": "max iterations reached" if g_inf_final > gtol else "",
        "history": history,
    }
    logger.info("Semi-Newton done nit=%d g_inf=%.6g", max_outer, g_inf_final)
    return P, info
