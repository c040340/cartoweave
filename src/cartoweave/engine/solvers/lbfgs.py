from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize

from ..core_eval import energy_and_grad_fullP
from ...utils.logging import logger


def solve_layout_lbfgs(scene, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels_init = scene.get("labels_init")
    N = 0 if labels_init is None else labels_init.shape[0]
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

    def _recorder(P, E, comps, meta):
        history["records"].append({
            "P": P.copy(),
            "E": float(E),
            "comps": {k: v.copy() for k, v in comps.items()},
            "meta": dict(meta) if meta else {},
        })

    E0, _, _ = energy_and_grad_fullP(scene, labels_init, cfg, record=_recorder)
    history["positions"].append(np.asarray(labels_init, float).copy())
    history["energies"].append(float(E0))
    last_E = E0

    def fun(x: np.ndarray):
        nonlocal last_E
        P = x.reshape(N, 2)
        E, G, _ = energy_and_grad_fullP(scene, P, cfg, record=_recorder)
        last_E = E
        return E, G.reshape(-1)

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
