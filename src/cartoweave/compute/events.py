from __future__ import annotations

from typing import Any, Dict, Tuple, Sequence


def _get(report: Any, keys: Sequence[str], default: Any) -> Any:
    for k in keys:
        if isinstance(report, dict) and k in report:
            return report[k]
        if hasattr(report, k):
            return getattr(report, k)
    return default


def report_to_event(report: dict | Any, algo_name: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Translate an optimizer-specific ``report`` into a standard event and metrics.

    Parameters
    ----------
    report:
        Structure produced by an optimizer iteration.  May be a ``dict`` or any
        object exposing attributes.
    algo_name:
        Human readable name of the optimizer (e.g. ``"LBFGS"``).
    """

    iter_in_algo = int(_get(report, ["iter_in_algo", "iter", "it", "k"], 0))
    step_size = float(_get(report, ["step_size", "alpha", "stepsize"], 0.0))
    ls_evals = int(_get(report, ["ls_evals", "ls_evaluations", "nfev"], 0))
    wolfe = _get(report, ["wolfe"], [])
    if isinstance(wolfe, Sequence) and not isinstance(wolfe, (str, bytes)):
        wolfe_list = list(wolfe)
    elif wolfe in (None, ""):
        wolfe_list = []
    else:
        wolfe_list = [bool(wolfe)]
    delta_E = float(_get(report, ["delta_E", "dE", "deltaE"], 0.0))
    gnorm = float(_get(report, ["g_norm", "gnorm", "g", "g_inf"], 0.0))
    g_inf = float(_get(report, ["g_inf"], gnorm))

    event = {
        "kind": "optimizer_step",
        "algo": algo_name,
        "iter_in_algo": iter_in_algo,
        "step_size": step_size,
        "ls_evals": ls_evals,
        "wolfe": wolfe_list,
        "delta_E": delta_E,
        "gnorm": gnorm,
    }

    metrics = {
        "g_norm": gnorm,
        "g_inf": g_inf,
        "step_size": step_size,
        "iters_in_algo": float(iter_in_algo),
    }

    return event, metrics


__all__ = ["report_to_event"]
