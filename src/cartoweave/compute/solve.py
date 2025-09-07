"""Action-driven solver entrypoint producing ViewPack v1."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from cartoweave.contracts.solvepack import SolvePack

from .events import report_to_event
from .eval import energy_and_grad_full
from .optim.loop import LoopContext as _EngineCtx, run_iters as _run_iters
from .passes import PassManager, get_pass_cfg
from .recorder import ViewRecorder
from .array_utils import expand_subset, expand_comps_subset
from .sources import make_sources_from_scene

logger = logging.getLogger(__name__)

Array2 = np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_state(pack: SolvePack) -> tuple[Array2, list[Any], np.ndarray]:
    """Return initial position, labels and active mask."""

    p = np.asarray(pack.P0, float).copy()
    labels = [deepcopy(lbl) for lbl in pack.labels0]
    active = np.asarray(pack.active0, bool)
    return p, labels, active


def _build_run_iters(pack: SolvePack):
    """Create a closure invoking the minimal gradient loop."""

    compute_cfg = pack.cfg.get("compute", {}) if isinstance(pack.cfg, dict) else {}

    def _run(
        p0: Array2,
        ctx: dict,
        energy_fn,
        iters_override: int | None = None,
        on_iter=None,
    ):
        mode = compute_cfg.get("solver", {}).get("public", {}).get("mode", "lbfgsb")
        tuning = compute_cfg.get("solver", {}).get("tuning", {})
        iters = iters_override or tuning.get(mode, {}).get("maxiter", 1)
        eng_ctx = _EngineCtx(
            labels=ctx["labels"],
            scene=ctx["scene"],
            active=ctx["active_ids"],
            cfg=compute_cfg,
            iters=int(iters),
            mode=mode,
            params={},
        )
        p_new, reports = _run_iters(p0, eng_ctx, energy_fn, report=True, on_iter=on_iter)
        E, G, comps = energy_fn(
            p_new, ctx["labels"], ctx["scene"], ctx["active_ids"], compute_cfg
        )
        gnorm = float(np.linalg.norm(G)) if G is not None and G.size else 0.0
        return p_new, {
            "E": E,
            "G": G,
            "gnorm": gnorm,
            "iters": reports[-1].it + 1 if reports else 0,
            "comps": comps,
            "mode": mode,
            "reports": reports,
        }

    return _run


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve(pack: SolvePack, *args, **kwargs):  # noqa: ARG001
    """Run optimization actions and assemble a :class:`ViewPack` v1."""

    actions = list(getattr(pack, "actions", []) or [])
    if not actions:
        raise ValueError("No actions provided in SolvePack; nothing to solve.")

    pm = kwargs.get("pass_manager") or kwargs.get("pm")
    if pm is None:
        pm = PassManager(pack.cfg.get("compute", {}), getattr(pack, "passes", None))

    pm.ensure_pass("action", position=0)
    pm.remove_pass("behavior")

    P_curr, labels, active = _init_state(pack)
    scene = pack.scene0.model_dump()
    energy_fn = pm.wrap_energy(energy_and_grad_full)
    step_fn = pm.wrap_step(lambda p_old, p_new, metrics: p_new)
    run_iters = _build_run_iters(pack)

    # ------------------------------------------------------------------
    # Recorder setup
    # ------------------------------------------------------------------
    recorder = ViewRecorder()

    N = len(pack.labels0)
    labels_meta = [
        lbl.model_dump() if hasattr(lbl, "model_dump") else dict(lbl)  # type: ignore[arg-type]
        for lbl in pack.labels0
    ]

    wh_rows = [getattr(lbl, "WH", None) for lbl in pack.labels0]
    WH = None
    if any(w is not None for w in wh_rows):
        WH = np.zeros((N, 2), float)
        for i, w in enumerate(wh_rows):
            WH[i] = (w if w is not None else (0.0, 0.0))

    sources = make_sources_from_scene(scene)

    defaults = {"compute": pack.cfg.get("compute", {})}
    aux: Dict[str, Any] = {"uid": pack.uid, "rng_seed": pack.rng_seed}
    recorder.start_run(N, labels_meta, WH, sources, defaults, aux)

    tuning = (
        ((getattr(pack, "cfg", {}) or {}).get("compute", {}) or {})
        .get("solver", {})
        .get("tuning", {})
        or {}
    )
    per_action_iters = int(((tuning.get("warmup", {}) or {}).get("steps", 1)) or 1)

    capture_cfg = get_pass_cfg(
        (pm.cfg if isinstance(pm.cfg, dict) else {}),
        "capture",
        {"every": 1, "limit": None, "final_always": True},
    )
    cap_every = max(1, int(capture_cfg.get("every", 1)))
    cap_limit = capture_cfg.get("limit")
    cap_final = bool(capture_cfg.get("final_always", True))

    t_global = 0
    P_prev_full = P_curr.copy()
    comps_prev_full: Dict[str, np.ndarray] = {}
    for pass_id, action in enumerate(actions):
        pass_name = getattr(action, "type", None) or getattr(action, "pass_name", None) or "action"
        recorder.start_pass(pass_id, pass_name)

        ctx = {
            "pack": pack,
            "scene": scene,
            "labels": labels,
            "active_ids": active,
            "P": P_curr,
            "step_index": pass_id,
        }
        pm.run_step(ctx)

        # update possibly mutated labels/active mask
        labels = ctx.get("labels", labels)
        active = np.asarray(ctx.get("active_ids", active), bool)
        active_idx = np.flatnonzero(active)

        _last_iter_recorded = -1

        def _on_iter(it: int, P_iter: np.ndarray, meta: Dict[str, Any]) -> None:
            nonlocal t_global, P_prev_full, comps_prev_full, _last_iter_recorded
            if cap_limit is not None and isinstance(cap_limit, int):
                if len(recorder.frames) >= cap_limit:
                    return
            if it % cap_every != 0:
                return
            P_full_i = expand_subset(P_prev_full, active_idx, P_iter[active_idx])
            comps_full_i = expand_comps_subset(
                comps_prev_full, active_idx, meta.get("comps", {})
            )
            meta_i: Dict[str, Any] = {
                "schema_version": "compute-v2",
                "pass_id": pass_id,
                "pass_name": pass_name,
                "frame_in_pass": it,
                "status": "ok",
                "events": pm.pop_events(),
            }
            gnorm_i = float(
                np.linalg.norm(meta.get("G"))
                if meta.get("G") is not None and meta.get("G").size
                else 0.0
            )
            metrics_i = {"E": float(meta.get("E", 0.0)), "gnorm": gnorm_i}
            recorder.record_frame(
                t=t_global,
                P_full=P_full_i,
                comps_full=comps_full_i,
                E=meta.get("E", 0.0),
                active_mask=active.copy(),
                meta_base=meta_i,
                metrics=metrics_i,
                field=None,
                G_snapshot=meta.get("G"),
            )
            P_prev_full = P_full_i
            comps_prev_full = comps_full_i
            _last_iter_recorded = it
            t_global += 1

        P_prop, metrics = run_iters(
            P_curr, ctx, energy_fn, iters_override=per_action_iters, on_iter=_on_iter
        )
        P_curr = step_fn(P_curr, P_prop, metrics)

        reports = metrics.pop("reports", [])
        event = None
        m_extra: Dict[str, float] = {}
        if reports:
            event, m_extra = report_to_event(reports[-1], metrics.get("mode", ""))

        metrics_all = {"E": float(metrics.get("E", 0.0)), "gnorm": float(metrics.get("gnorm", 0.0))}
        metrics_all.update({k: float(v) for k, v in m_extra.items()})

        meta_base: Dict[str, Any] = {
            "schema_version": "compute-v2",
            "pass_id": pass_id,
            "pass_name": pass_name,
            "frame_in_pass": "final",
            "status": "ok",
            "reason": "",
            "timings_ms": {},
            "events": pm.pop_events(),
        }
        if event is not None:
            meta_base["optimizer_step"] = event

        P_full = expand_subset(P_prev_full, active_idx, P_curr[active_idx])
        comps_full = expand_comps_subset(
            comps_prev_full, active_idx, metrics.get("comps", {})
        )

        iters = int(metrics.get("iters", len(reports)))
        need_final = cap_final and (_last_iter_recorded != (iters - 1))
        if cap_limit is not None and isinstance(cap_limit, int):
            if len(recorder.frames) >= cap_limit:
                need_final = False
        if need_final:
            metrics_all_final = metrics_all
            recorder.record_frame(
                t=t_global,
                P_full=P_full,
                comps_full=comps_full,
                E=metrics.get("E", 0.0),
                active_mask=active.copy(),
                meta_base=meta_base,
                metrics=metrics_all_final,
                field=None,
                G_snapshot=metrics.get("G"),
            )
            t_global += 1
        else:
            # discard unused events
            pm.pop_events()
            if event is not None and recorder.frames:
                last = recorder.frames[-1]
                last.meta.setdefault("events", []).append(
                    {
                        "kind": "optimizer_step",
                        "algo": event.get("algo"),
                        "iter_in_algo": event.get("iter_in_algo"),
                        "step_size": event.get("step_size"),
                        "ls_evals": event.get("ls_evals"),
                        "wolfe": event.get("wolfe"),
                        "delta_E": event.get("delta_E"),
                        "gnorm": event.get("gnorm"),
                    }
                )

        P_prev_full = P_full
        comps_prev_full = comps_full

        recorder.end_pass()

    vp = recorder.finish()
    vp.validate()
    return vp


__all__ = ["solve"]

