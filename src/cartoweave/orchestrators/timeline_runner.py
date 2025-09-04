"""Minimal timeline runner with per-action recording.

This helper is intentionally lightweight and only implements the features
required by the test-suite.  Each action simply invokes the L-BFGS solver while
tagging all recording callbacks with the corresponding ``action_id`` and
``action_name``.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List

import numpy as np

from cartoweave.engine.solvers import lbfgs


def run_timeline(
    scene: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
):
    """Execute a sequence of actions with per-action recording.

    Parameters
    ----------
    scene:
        Base scene description shared by all actions.
    timeline:
        List of action dictionaries.  Only the ``name`` field is inspected here.
    cfg:
        Solver configuration shared by all actions.  Defaults to ``{}``.
    record:
        Optional callback receiving per-evaluation snapshots.  The callback is
        augmented with ``action_id`` and ``action_name`` tags.
    """

    cfg = cfg or {}
    P_cur = np.asarray(scene.get("labels_init", np.zeros((0, 2), float)), float)
    infos: List[Dict[str, Any]] = []

    for a_idx, action in enumerate(timeline):
        def _rec_with_action(P, E, comps, meta):
            m = {"action_id": a_idx, "action_name": action.get("name", f"action_{a_idx}")}
            if meta:
                m.update(meta)
            if record:
                record(P, E, comps, m)

        info = lbfgs.run(scene, P_cur, cfg, record=_rec_with_action)
        P_cur = info.get("P", P_cur)
        infos.append(info)

    return {"actions": infos, "P_final": P_cur}

