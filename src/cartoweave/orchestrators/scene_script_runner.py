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


def _build_active_views_for_solver(scene: Dict[str, Any]) -> List[int]:
    """Slice per-label arrays to the solver's active subset.

    The scene is expected to provide ``*_all`` arrays from which the active
    views are derived.  Sliced arrays are written back without the suffix so
    downstream solvers operate only on the subset.  The list of active
    indices is returned unchanged for convenience.
    """

    ids = scene.get("_active_ids_solver", [])
    scene["labels_init"] = np.asarray(scene["labels_init_all"], float)[ids]
    scene["WH"] = np.asarray(scene["WH_all"], float)[ids]
    if "anchors_all" in scene and scene["anchors_all"] is not None:
        scene["anchors"] = np.asarray(scene["anchors_all"], float)[ids]
    # Additional per-label arrays used by terms can be sliced here as needed.
    return ids


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

        # Restore full arrays so ``apply_step_to_scene`` can operate on them.
        scene["labels"] = labels_all
        scene["labels_init_all"] = P_all
        scene["WH_all"] = WH_all
        scene["WH"] = WH_all
        if anchors_all is not None:
            scene["anchors_all"] = anchors_all
            scene["anchors"] = anchors_all

        apply_step_to_scene(scene, step)

        active_ids_solver = [
            i
            for i, l in enumerate(scene["labels"])
            if l.get("visible", False) and l.get("mode") != "circle"
        ]
        active_ids_viz = [
            i for i, l in enumerate(scene["labels"]) if l.get("visible", False)
        ]
        scene["_active_ids_solver"] = active_ids_solver
        scene["_active_ids_viz"] = active_ids_viz
        scene["_current_step_name"] = step.get("name")

        if step_idx == 0:
            for lab in scene["labels"]:
                lab["visible"] = lab.get("visible", False) and False
            apply_step_to_scene(scene, step)
            active_ids_solver = [
                i
                for i, l in enumerate(scene["labels"])
                if l.get("visible", False) and l.get("mode") != "circle"
            ]
            active_ids_viz = [
                i for i, l in enumerate(scene["labels"]) if l.get("visible", False)
            ]
            scene["_active_ids_solver"] = active_ids_solver
            scene["_active_ids_viz"] = active_ids_viz

        ids = _build_active_views_for_solver(scene)

        active_ids = [
            i for i, l in enumerate(scene["labels"]) if l.get("visible") and l.get("mode") != "circle"
        ]
        scene["_active_ids"] = active_ids
        print(
            f"[runner] step={step.get('name')} active_labels={len(active_ids)} ids={active_ids[:8]}..."
        )
        assert len(active_ids) == scene["labels_init"].shape[0], (
            f"Active mismatch before solve: {len(active_ids)} vs P {scene['labels_init'].shape}"
        )

        scene["_log_label_stats"] = True
        P_active = scene.get("labels_init", np.zeros((0, 2), float))
        P_pre = P_all.copy()

        info = run_solve_plan(scene, stages, cfg)

        P_active = np.asarray(info.get("P_final", P_active), float)
        if P_all.shape[0] >= len(ids) and len(ids) > 0:
            P_all[ids] = P_active
        scene["labels_init"] = P_active
        scene["labels_init_all"] = P_all

        hist = info.get("history", {})
        pos = list(hist.get("positions", []))
        eng = list(hist.get("energies", []))
        rec = list(hist.get("records", []))

        if history_pos and pos:
            pos = pos[1:]
            eng = eng[1:]
            rec = rec[1:]

        pos_full = []
        for arr in pos:
            arr = np.asarray(arr, float)
            if arr.shape[0] == len(ids):
                full = np.full_like(P_pre, np.nan)
                full[ids] = arr
                pos_full.append(full)
            else:
                pos_full.append(arr)
        pos = pos_full

        for r in rec:
            P_snap = np.asarray(r.get("P"), float)
            if P_snap.shape[0] == len(ids):
                full = np.full_like(P_pre, np.nan)
                full[ids] = P_snap
                r["P"] = full

            comps = r.get("comps")
            if isinstance(comps, dict):
                comps_full: Dict[str, np.ndarray] = {}
                for k, arr in comps.items():
                    arr_np = np.asarray(arr, float)
                    if arr_np.shape[0] == len(ids):
                        full = np.zeros((P_pre.shape[0], 2), float)
                        full[ids] = arr_np
                        comps_full[k] = full
                    else:
                        comps_full[k] = arr_np
                r["comps"] = comps_full

            meta = r.setdefault("meta", {})
            meta.setdefault("step_id", step_idx)
            meta.setdefault("step_name", step.get("name", f"step_{step_idx}"))

        history_pos.extend(pos)
        history_E.extend(eng)
        history_rec.extend(rec)

        rec_end = len(history_rec)
        history_steps.append(
            {
                "name": step.get("name", f"step_{step_idx}"),
                "rec_start": rec_start,
                "rec_end": rec_end,
                "active_ids_solver": list(active_ids_solver),
                "active_ids_viz": list(active_ids_viz),
            }
        )

        scene["_current_step_name"] = None

    scene["labels"] = labels_all
    scene["labels_init"] = P_all
    scene["labels_init_all"] = P_all
    scene["WH"] = WH_all
    scene["WH_all"] = WH_all
    if anchors_all is not None:
        scene["anchors"] = anchors_all
        scene["anchors_all"] = anchors_all

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

    return {"stages": stages_meta, "P_final": P_all, "history": history}

