from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np
from scipy.optimize import minimize

from ...utils.logging import logger


def _cfg(cfg: Dict[str, Any], key: str, default):
    """Read ``key`` from ``cfg`` without raising on missing entries."""
    return cfg[key] if key in cfg else default


def solve_layout_lbfgs(
    scene,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
    energy_fn: Callable | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels_init = np.asarray(scene.get("labels_init"), float)
    assert labels_init.ndim == 2 and labels_init.shape[1] == 2, (
        f"P must be (N,2), got {labels_init.shape}"
    )
    WH = np.asarray(scene.get("WH", np.zeros((labels_init.shape[0], 2))), float)
    active = scene.get("_active_ids_solver")
    active_len = labels_init.shape[0] if active is None else len(active)
    assert labels_init.shape[0] == WH.shape[0] == active_len, (
        f"Solver shape mismatch: P={labels_init.shape}, WH={WH.shape}, active={active_len}"
    )
    N = labels_init.shape[0]
    logger.info("L-BFGS start n_labels=%d", N)
    if N == 0:
        logger.info("L-BFGS early exit: no labels")
        return np.zeros((0, 2), dtype=float), {
            "nit": 0,
            "nfev": 0,
            "msg": "no labels",
            "history": {"positions": [], "energies": [], "records": []},
        }

    x0 = labels_init.reshape(-1).astype(float)

    history = {"positions": [], "energies": [], "records": []}
    _eval_counter = {"n": 0}

    def _recorder(P, E, comps, meta):
        history["records"].append(
            {
                "P": np.asarray(P, float).copy(),
                "E": float(E),
                "comps": {k: np.asarray(v, float).copy() for k, v in comps.items()},
                "meta": dict(meta) if meta else {},
                "eval_index": _eval_counter["n"],
            }
        )
        if record is not None:
            record(P, E, comps, meta)

    pgtol = float(_cfg(cfg, "lbfgs_pgtol", 1e-3))
    maxiter = int(_cfg(cfg, "lbfgs_maxiter", 150))

    if energy_fn is None:
        from ...compute.eval import energy_and_grad_full as _eval

        def energy_fn(sc, P, cfg_, record=None):
            mask = np.ones(P.shape[0], bool)
            E, G, comps, _ = _eval(P, sc, mask, cfg_)
            if record is not None:
                record(P, E, comps, {})
            return E, G

    E0, G0 = energy_fn(scene, labels_init, cfg, record=_recorder)
    g0 = float(np.linalg.norm(G0, np.inf))
    _eval_counter["n"] += 1
    history["positions"].append(np.asarray(labels_init, float).copy())
    history["energies"].append(float(E0))
    last_E = E0
    meta0 = history["records"][-1].setdefault("meta", {})
    meta0["solver_info"] = {
        "solver": "lbfgs",
        "g_inf": g0,
        "gtol": pgtol,
        "iter": 0,
        "iter_max": maxiter,
    }

    last_g = g0

    def fun(x: np.ndarray):
        nonlocal last_E, last_g
        P = x.reshape(N, 2)
        E, G = energy_fn(scene, P, cfg, record=_recorder)
        g = float(np.linalg.norm(G, np.inf))
        _eval_counter["n"] += 1
        meta = history["records"][-1].setdefault("meta", {})
        meta["solver_info"] = {
            "solver": "lbfgs",
            "g_inf": g,
            "gtol": pgtol,
            "iter": _eval_counter["n"],
            "iter_max": maxiter,
        }
        last_E = E
        last_g = g
        return E, G.reshape(-1).astype(float)

    def callback(xk: np.ndarray):
        P = xk.reshape(N, 2)
        history["positions"].append(P.copy())
        history["energies"].append(float(last_E))
        logger.debug("L-BFGS iter %d E=%.6g", len(history["energies"]) - 1, float(last_E))

    res = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": maxiter, "gtol": pgtol},
    )
    P_opt = res.x.reshape(N, 2)
    info = {
        "nit": res.nit,
        "nfev": res.nfev,
        "msg": res.message,
        "history": history,
        "g_inf": last_g,
        "gtol": pgtol,
    }
    logger.info("L-BFGS done nit=%s nfev=%s", res.nit, res.nfev)
    return P_opt, info


def run(
    scene: Dict[str, Any],
    P0: np.ndarray,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
    energy_fn: Callable | None = None,
    **kw,
) -> Dict[str, Any]:
    """Convenience wrapper used by tests and timeline orchestrators.

    Parameters
    ----------
    scene:
        Scene dictionary describing forces and geometry.
    P0:
        Initial label positions ``(N,2)``.
    cfg:
        Solver configuration dictionary.
    record:
        Optional callback invoked on every energy/gradient evaluation.
    """

    sc = dict(scene)
    sc["labels_init"] = np.asarray(P0, float)
    P_opt, info = solve_layout_lbfgs(sc, cfg, record=record, energy_fn=energy_fn)
    info["P"] = P_opt
    return info
