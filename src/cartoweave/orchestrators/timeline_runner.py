# NOTE:
#     The original example invoked the L-BFGS solver directly without
#     threading action metadata through the recording callbacks, leaving
#     ``history.records`` empty of ``action_id``/``action_name`` tags and the UI
#     unable to display the action bar.  This runner now forwards the recorder
#     and patches history entries so each evaluation is associated with its
#     originating action.

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
        # Wrap the recorder so downstream solvers tag each evaluation with the
        # current action metadata.  ``lbfgs.run`` ignores ``record=None`` so we
        # explicitly forward the wrapper to ensure real-time callbacks are
        # invoked when provided.
        def _rec(P, E, comps, meta):
            meta = dict(meta) if meta else {}
            meta.setdefault("action_id", a_idx)
            meta.setdefault("action_name", action.get("name", f"action_{a_idx}"))
            if record:
                record(P, E, comps, meta)

        info = lbfgs.run(scene, P_cur, cfg, record=_rec)
        P_cur = info.get("P", P_cur)

        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))

        # ``_rec`` only affects the live callback; solver history must be
        # patched post-hoc so every record carries action information.
        for r in rec:
            meta = r.setdefault("meta", {})
            meta.setdefault("action_id", a_idx)
            meta.setdefault("action_name", action.get("name", f"action_{a_idx}"))

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

