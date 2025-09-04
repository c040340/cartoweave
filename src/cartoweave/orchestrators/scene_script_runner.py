from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from .solve_plan_runner import run_solve_plan


def apply_step_to_scene(scene: Dict[str, Any], step: Dict[str, Any]) -> None:
    """Apply a single scene-script step to ``scene`` in-place.

    Parameters
    ----------
    scene:
        Scene dictionary containing ``labels`` and ``WH`` arrays.
    step:
        Dictionary describing the step. Expected keys include:

        - ``op``: one of ``"enter"``, ``"mode"`` or ``"disappear"``.
        - ``id``: label identifier.
        - ``mode``: optional new mode when ``op`` is ``"enter"`` or ``"mode"``.
        - ``wh``: optional ``(w, h)`` tuple supplying explicit size for the new
          mode.  When absent, per-label ``modes`` specs are consulted.

    The function mutates ``scene`` directly and performs **no** solving.
    """

    if not scene or not step:
        return

    labels = scene.get("labels", [])
    label_id = step.get("id")
    if label_id is None:
        return

    idx = next((i for i, lab in enumerate(labels) if lab.get("id") == label_id), None)
    if idx is None:
        return

    label = labels[idx]
    op = step.get("op")
    if op in ("appear", "enter"):
        op = "enter"
    elif op in ("change", "mode"):
        op = "mode"
    elif op in ("hide", "disappear"):
        op = "disappear"

    def _update_wh(new_mode: str) -> None:
        wh = step.get("wh")
        if wh is None:
            modes = label.get("modes", {})
            spec = modes.get(new_mode)
            if isinstance(spec, dict):
                w = spec.get("w", scene["WH"][idx, 0])
                h = spec.get("h", scene["WH"][idx, 1])
                wh = (w, h)
        if wh is not None:
            arr = np.asarray(wh, float).reshape(2,)
            scene.setdefault("WH", np.zeros((len(labels), 2), float))
            scene["WH"][idx] = arr

    if op == "enter":
        label["visible"] = True
        new_mode = step.get("mode")
        if new_mode is not None:
            label["mode"] = new_mode
            _update_wh(new_mode)
    elif op == "mode":
        new_mode = step.get("mode")
        if new_mode is None:
            return
        label["mode"] = new_mode
        _update_wh(new_mode)
    elif op == "disappear":
        label["visible"] = False


def run_scene_script(
    scene: Dict[str, Any],
    scene_script: List[Dict[str, Any]],
    solve_plan: Dict[str, Any] | List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
):
    """Execute a scene script with a fixed solve plan.

    Each step mutates ``scene`` via :func:`apply_step_to_scene` and then runs
    all stages of ``solve_plan``.  Solver history is concatenated across steps
    and annotated with ``step_id``/``step_name`` metadata.  The returned
    history additionally contains ``scene_steps`` entries describing the record
    ranges contributed by each step.
    """

    cfg = cfg or {}
    steps = list(scene_script or [])
    stages = (
        list(solve_plan.get("stages", []))
        if isinstance(solve_plan, dict)
        else list(solve_plan or [])
    )

    history_pos: List[np.ndarray] = []
    history_E: List[float] = []
    history_rec: List[Dict[str, Any]] = []
    history_steps: List[Dict[str, Any]] = []

    # Preserve full per-label arrays.  During solving, only the active subset is
    # exposed via keys without the ``_all`` suffix.  ``labels`` need special
    # handling as they are a list, not an ``ndarray``.
    labels_all = scene.get("labels", [])
    scene["labels_all"] = labels_all

    N_total = len(labels_all)

    def _ensure_all(key: str) -> np.ndarray:
        arr = scene.get(f"{key}_all")
        if arr is None:
            arr = np.asarray(scene.get(key, np.zeros((N_total, 2), float)), float)
            scene[f"{key}_all"] = arr
        return np.asarray(arr, float)

    P_all = _ensure_all("labels_init")
    WH_all = _ensure_all("WH")
    anchors_all = scene.get("anchors_all")
    if anchors_all is None and scene.get("anchors") is not None:
        anchors_all = np.asarray(scene["anchors"], float)
        scene["anchors_all"] = anchors_all

    P_active = np.zeros((0, 2), float)

    def _canon(op: str | None) -> str | None:
        if op in ("appear", "enter"):
            return "enter"
        if op in ("change", "mode"):
            return "mode"
        if op in ("hide", "disappear"):
            return "disappear"
        return op

    for step_idx, step in enumerate(steps):
        rec_start = len(history_rec)

        step_name = step.get("name", f"step_{step_idx}")
        scene["_current_step_name"] = step_name

        # Restore full arrays so ``apply_step_to_scene`` can operate on them.
        scene["labels"] = labels_all
        scene["WH"] = WH_all
        if anchors_all is not None:
            scene["anchors"] = anchors_all
        scene["labels_init"] = P_all

        apply_step_to_scene(scene, step)

        # Determine active labels (visible only). Circle labels are retained and
        # skipped inside individual terms as needed.
        _active_ids = [
            i for i, lab in enumerate(labels_all) if lab.get("visible", True)
        ]

        scene["_active_ids"] = _active_ids

        # Slice per-label arrays for the solver. ``labels`` remains the full
        # list so force terms can reference original indices via
        # ``scene['_active_ids']``.
        for k, v in list(scene.items()):
            if k.endswith("_all") and isinstance(v, np.ndarray):
                scene[k[:-4]] = np.asarray(v, float)[_active_ids]

        scene["_log_label_stats"] = True
        P_active = scene.get("labels_init", np.zeros((0, 2), float))

        # Keep a full-scene snapshot so per-iteration records can be expanded to
        # a consistent shape even when the active subset changes.
        P_pre = P_all.copy()

        info = run_solve_plan(scene, stages, cfg)

        P_active = np.asarray(info.get("P_final", P_active), float)

        # Update the full position array with the solved subset.
        if P_all.shape[0] >= len(_active_ids) and len(_active_ids) > 0:
            P_all[_active_ids] = P_active
        scene["labels_init"] = P_active

        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))

        if history_pos and pos:
            pos = pos[1:]
            eng = eng[1:]
            rec = rec[1:]

        # Expand per-step history to full label count so trajectories can be
        # stacked across steps.
        pos_full = []
        for arr in pos:
            arr = np.asarray(arr, float)
            if arr.shape[0] == len(_active_ids):
                full = P_pre.copy()
                full[_active_ids] = arr
                pos_full.append(full)
            else:
                pos_full.append(arr)
        pos = pos_full

        for r in rec:
            P_snap = np.asarray(r.get("P"), float)
            if P_snap.shape[0] == len(_active_ids):
                full = P_pre.copy()
                full[_active_ids] = P_snap
                r["P"] = full
            meta = r.setdefault("meta", {})
            meta.setdefault("step_id", step_idx)
            meta.setdefault("step_name", step.get("name", f"step_{step_idx}"))

        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)

        rec_end = len(history_rec)
        entry = {
            "name": step.get("name", f"step_{step_idx}"),
            "rec_start": rec_start,
            "rec_end": rec_end,
        }
        op = _canon(step.get("op"))
        if op:
            entry[op] = {k: v for k, v in step.items() if k not in {"name", "op"}}
        history_steps.append(entry)

        scene.pop("_current_step_name", None)

    history = {
        "positions": history_pos,
        "energies": history_E,
        "records": history_rec,
        "scene_steps": history_steps,
    }

    stages_meta = [
        {"name": st.get("name", f"stage_{i}")}
        for i, st in enumerate(stages)
    ]

    return {"stages": stages_meta, "P_final": P_active, "history": history}

