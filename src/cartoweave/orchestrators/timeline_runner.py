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
    actions_meta: List[Dict[str, Any]] = []

    history_pos: List[np.ndarray] = []
    history_E: List[float] = []
    history_rec: List[Dict[str, Any]] = []

    for a_idx, action in enumerate(timeline):
        info = lbfgs.run(scene, P_cur, cfg, record=None)
        P_cur = info.get("P", P_cur)

        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))

        for r in rec:
            meta = r.setdefault("meta", {})
            meta.setdefault("action_id", a_idx)
            meta.setdefault("action_name", action.get("name", f"action_{a_idx}"))
            if record:
                record(r.get("P"), r.get("E"), r.get("comps", {}), meta)

        if history_pos:
            if pos:
                pos = pos[1:]
                eng = eng[1:]
                rec = rec[1:]

        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)

        actions_meta.append({"name": action.get("name", f"action_{a_idx}")})

    history = {"positions": history_pos, "energies": history_E, "records": history_rec}
    return {"actions": actions_meta, "P_final": P_cur, "history": history}

