# -*- coding: utf-8 -*-
"""Adapter for calling solvers from the engine package.

The compute module assembles energies and gradients but delegates the
iteration loop to existing solvers in :mod:`cartoweave.engine.solvers`.
This module provides a thin wrapper that normalizes parameters and wiring
so that compute's :func:`~cartoweave.compute.run.solve` can invoke the
engine solvers in a uniform way.
"""

from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np


def _import_solver(mode: str):
    if mode == "lbfgs":
        from cartoweave.engine.solvers import lbfgs as _impl
    elif mode in ("semi", "semi_newton", "seminewton"):
        from cartoweave.engine.solvers import semi_newton as _impl
    else:
        raise ValueError(f"Unknown solver mode: {mode}")
    return _impl


def run_via_engine_solver(
    mode: str,
    *,
    P0,
    scene,
    active_mask,
    params: Dict[str, Any],
    energy_and_grad: Callable,
    recorder: Callable[[Dict], None],
) -> Dict[str, Any]:
    """Call an engine solver with compute's energy callback.

    Parameters
    ----------
    mode:
        Solver name (``"lbfgs"`` or ``"semi"``).
    P0, scene, active_mask, params:
        Initial positions, immutable scene, active mask and solver
        parameters passed through unchanged.
    energy_and_grad:
        Callable returning ``(E, G, comps, meta)`` following compute's
        conventions.
    recorder:
        Callback invoked after each evaluation with a dict containing
        ``P``, ``G``, ``comps`` and ``E``.
    """
    impl = _import_solver(mode)
    active_ids = np.nonzero(active_mask)[0]

    def energy_fn(scene_, P, cfg_, record=None):
        E, G, comps, meta = energy_and_grad(P, scene_, active_mask, cfg_)
        if record is not None:
            record(P, E, comps, meta)
        recorder({"P": P, "G": G, "comps": comps, "E": E, "mask": active_mask, "meta": meta})
        return E, G

    sc = dict(scene)
    sc["labels_init"] = P0

    if mode == "lbfgs":
        sc["_active_ids_solver"] = active_ids
        info = impl.run(scene=sc, P0=P0, cfg=params, record=None, energy_fn=energy_fn)
        history = info.get("history", {})
        energies = history.get("energies", [])
        result = {
            "P_final": info.get("P"),
            "iters": info.get("nit"),
            "stop_reason": info.get("msg", info.get("message", "")),
            "converged": info.get("g_inf", np.inf) <= info.get("gtol", np.inf),
            "E_last": energies[-1] if energies else None,
        }
    else:
        sc["movable_idx"] = active_ids
        P_final, info = impl.solve_layout_semi_newton(sc, params, record=None, energy_fn=energy_fn)
        history = info.get("history", {})
        energies = history.get("energies", [])
        result = {
            "P_final": P_final,
            "iters": info.get("nit"),
            "stop_reason": info.get("message", info.get("msg", "")),
            "converged": info.get("success"),
            "E_last": energies[-1] if energies else None,
        }

    return result


# Backward compatibility ----------------------------------------------------

# DEPRECATED: to be removed in 2 releases
run_via_legacy_solver = run_via_engine_solver
