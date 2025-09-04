# NOTE:
#     The original example invoked the L-BFGS solver directly without
#     threading stage metadata through the recording callbacks, leaving
#     ``history.records`` empty of ``stage_id``/``stage_name`` tags and the UI
#     unable to display the action bar.  This runner now forwards the recorder
#     and patches history entries so each evaluation is associated with its
#     originating stage.

"""Minimal solve-plan runner with per-stage recording.

This helper is intentionally lightweight and only implements the features
    required by the test-suite.  Each stage simply invokes the L-BFGS solver while
    tagging all recording callbacks with the corresponding ``stage_id`` and
    ``stage_name``.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, List

import numpy as np

from cartoweave.engine.solvers import lbfgs


def run_solve_plan(
    scene: Dict[str, Any],
    solve_plan: List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
    record: Callable[[np.ndarray, float, Dict[str, np.ndarray], Dict[str, Any]], None]
    | None = None,
):
    """Execute a sequence of stages with per-stage recording.

    Parameters
    ----------
    scene:
        Base scene description shared by all stages.
    solve_plan:
        List of stage dictionaries.  Only the ``name`` field is inspected here.
    cfg:
        Solver configuration shared by all stages.  Defaults to ``{}``.
    record:
        Optional callback receiving per-evaluation snapshots.  The callback is
        augmented with ``stage_id`` and ``stage_name`` tags.
    """

    cfg = cfg or {}
    if not solve_plan:
        raise ValueError("run_solve_plan received empty solve_plan")
    P_cur = np.asarray(scene.get("labels_init", np.zeros((0, 2), float)), float)
    stages_meta: List[Dict[str, Any]] = []

    history_pos: List[np.ndarray] = []
    history_E: List[float] = []
    history_rec: List[Dict[str, Any]] = []

    for stage_idx, stage in enumerate(solve_plan):
        stage_name = stage.get("name", f"stage_{stage_idx}")
        scene["_current_stage_name"] = stage_name
        # Wrap the recorder so downstream solvers tag each evaluation with the
        # current stage metadata.  ``lbfgs.run`` ignores ``record=None`` so we
        # explicitly forward the wrapper to ensure real-time callbacks are
        # invoked when provided.
        def _rec(P, E, comps, meta):
            meta = dict(meta) if meta else {}
            meta.setdefault("stage_id", stage_idx)
            meta.setdefault("stage_name", stage_name)
            if record:
                record(P, E, comps, meta)

        info = lbfgs.run(scene, P_cur, cfg, record=_rec)
        P_cur = info.get("P", P_cur)

        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))

        # ``_rec`` only affects the live callback; solver history must be
        # patched post-hoc so every record carries stage information.
        for r in rec:
            meta = r.setdefault("meta", {})
            meta.setdefault("stage_id", stage_idx)
            meta.setdefault("stage_name", stage_name)

        if history_pos:
            if pos:
                pos = pos[1:]
                eng = eng[1:]
                rec = rec[1:]

        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)

        stages_meta.append({"name": stage_name})

        scene["_current_stage_name"] = None

    history = {"positions": history_pos, "energies": history_E, "records": history_rec}
    return {"stages": stages_meta, "P_final": P_cur, "history": history}

