from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

from ..layout_utils import is_circle_label

# Colours used to style ratios in the info panel
_RATIO_COLORS = {
    "green": "#1f9d55",
    "amber": "#d69e2e",
    "red": "#e53e3e",
}

# Small epsilon to avoid division by zero
_EPS = 1e-12

# Previous absolute forces for delta computations
_prev_absF: Optional[np.ndarray] = None


def style_by_ratio(r: float) -> Dict[str, object]:
    """Return a colour/bold style mapping based on ``r``.

    Parameters
    ----------
    r:
        Ratio value.  Smaller or equal to one is considered good (green and
        bold).  Values up to three are amber and anything above red.
    """

    if r <= 1.0:
        return {"color": _RATIO_COLORS["green"], "bold": True}
    if r <= 3.0:
        return {"color": _RATIO_COLORS["amber"], "bold": False}
    return {"color": _RATIO_COLORS["red"], "bold": False}


def _cfg_get(cfg, name: str, default: float) -> float:
    """Helper to read ``name`` from ``cfg`` supporting both dict and objects."""

    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return float(cfg.get(name, default))
    return float(getattr(cfg, name, default))


def collect_solver_metrics(
    P_now,
    forces_per_label,
    per_term_forces,
    label_kinds,
    solver_info,
    cfg,
) -> Dict[str, object]:
    """Collect and aggregate various solver metrics for visualisation.

    Parameters
    ----------
    P_now : ``(N, 2)`` array-like
        Current label positions.  Only the length ``N`` is used.
    forces_per_label : ``(N, 2)`` array-like
        Net force vector acting on each label.
    per_term_forces : mapping, optional
        Individual force component arrays keyed by name.
    label_kinds : sequence of dict
        Label metadata describing whether entries are circles or locked.
    solver_info : dict
        Information supplied by the active solver stage (contains ``g_inf``,
        ``gtol`` and others).
    cfg : object or mapping
        Configuration storing e.g. ``viz_F_tol`` or ``force_stop_tol``.

    Returns
    -------
    dict
        Structure described in the user story – includes per label forces,
        global aggregates and formatting hints for the UI.
    """

    global _prev_absF

    forces = np.asarray(forces_per_label, dtype=float).reshape(-1, 2)
    N = forces.shape[0]

    absF = np.linalg.norm(forces, axis=1)

    if _prev_absF is None or len(_prev_absF) != N:
        deltaF = np.zeros(N, dtype=float)
        deltaF_over_F = np.zeros(N, dtype=float)
    else:
        deltaF = absF - _prev_absF
        deltaF_over_F = deltaF / np.clip(_prev_absF, _EPS, None)
    _prev_absF = absF.copy()

    # Movable mask excludes circles and locked labels
    movable = np.asarray(
        [not is_circle_label(lab) and not lab.get("locked", False) for lab in label_kinds],
        dtype=bool,
    )
    if movable.size < N:
        movable = np.pad(movable, (0, N - movable.size), constant_values=False)

    # Global infinity norm of forces over movable labels
    force_inf_per_label = np.max(np.abs(forces), axis=1)
    F_inf = float(np.max(force_inf_per_label[movable])) if np.any(movable) else 0.0

    # Force tolerance from configuration
    F_tol = _cfg_get(cfg, "viz_F_tol", _cfg_get(cfg, "force_stop_tol", 1e-3))

    # Gradient metrics from solver_info
    g_inf = float(solver_info.get("g_inf", 0.0))
    gtol = float(solver_info.get("gtol", 1.0))

    F_ratio = F_inf / max(F_tol, _EPS)
    g_ratio = g_inf / max(gtol, _EPS)

    F_style = style_by_ratio(F_ratio)
    g_style = style_by_ratio(g_ratio)

    topk = int(_cfg_get(cfg, "viz_topk", 3))
    top_force_labels = sorted(
        ((i, float(absF[i])) for i in range(N) if movable[i]),
        key=lambda x: x[1],
        reverse=True,
    )[:topk]

    out: Dict[str, object] = {
        "F_i": forces,
        "per_term": {k: np.asarray(v, float) for k, v in (per_term_forces or {}).items()},
        "F_inf": F_inf,
        "F_tol": F_tol,
        "g_inf": g_inf,
        "gtol": gtol,
        "top_force_labels": top_force_labels,
        "deltaF": deltaF,
        "deltaF_over_F": deltaF_over_F,
        "ui": {
            "F_ratio": F_ratio,
            "g_ratio": g_ratio,
            "F_style": F_style,
            "g_style": g_style,
        },
        # Pass through solver stage information for convenience
        "solver": solver_info.get("solver"),
        "iter": solver_info.get("iter"),
        "iter_max": solver_info.get("iter_max"),
        "stage": solver_info.get("stage"),
    }

    return out


__all__ = ["collect_solver_metrics", "style_by_ratio"]
