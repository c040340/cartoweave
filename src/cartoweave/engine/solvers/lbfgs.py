from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize

from ..core_eval import energy_and_grad_fullP

def solve_layout_lbfgs(scene, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels_init = scene.get("labels_init")
    N = 0 if labels_init is None else labels_init.shape[0]
    if N == 0:
        return np.zeros((0,2), dtype=float), {"nit": 0, "nfev": 0, "msg": "no labels"}

    x0 = labels_init.reshape(-1).astype(float)

    def fun(x: np.ndarray):
        P = x.reshape(N, 2)
        E, G, S = energy_and_grad_fullP(scene, P, cfg)
        return E, G.reshape(-1)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B")
    P_opt = res.x.reshape(N, 2)
    info = {"nit": res.nit, "nfev": res.nfev, "msg": res.message}
    return P_opt, info
