from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np
from scipy.optimize import minimize

from ..core_eval import energy_and_grad_fullP
from ...utils.logging import logger


def solve_layout_lbfgs(
    scene,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
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

    E0, _, _ = energy_and_grad_fullP(scene, labels_init, cfg, record=_recorder)
    _eval_counter["n"] += 1
    history["positions"].append(np.asarray(labels_init, float).copy())
    history["energies"].append(float(E0))
    last_E = E0

    def fun(x: np.ndarray):
        nonlocal last_E
        P = x.reshape(N, 2)
        E, G, _ = energy_and_grad_fullP(scene, P, cfg, record=_recorder)
        _eval_counter["n"] += 1
        last_E = E
        return E, G.reshape(-1).astype(float)

    def callback(xk: np.ndarray):
        P = xk.reshape(N, 2)
        history["positions"].append(P.copy())
        history["energies"].append(float(last_E))
        logger.debug("L-BFGS iter %d E=%.6g", len(history["energies"]) - 1, float(last_E))

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback)
    P_opt = res.x.reshape(N, 2)
    info = {"nit": res.nit, "nfev": res.nfev, "msg": res.message, "history": history}
    logger.info("L-BFGS done nit=%s nfev=%s", res.nit, res.nfev)
    return P_opt, info


def run(
    scene: Dict[str, Any],
    P0: np.ndarray,
    cfg: Dict[str, Any],
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
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
    P_opt, info = solve_layout_lbfgs(sc, cfg, record=record)
    info["P"] = P_opt
    return info
