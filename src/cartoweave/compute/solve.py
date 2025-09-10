"""Action-driven solver entrypoint producing ViewPack v1."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict

import numpy as np

from cartoweave.contracts.solvepack import SolvePack

from .array_utils import expand_comps_subset, expand_subset
from .eval import energy_and_grad_full
from .events import report_to_event
from .anchor_utils import label_anchor_xy
from .optim.loop import LoopContext as _EngineCtx
from .optim.loop import run_iters as _run_iters
from .passes import PassManager, get_pass_cfg
from .recorder import ViewRecorder
from .sources import make_sources_from_scene

logger = logging.getLogger("cartoweave.compute.solve")

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

    comp = (pack.cfg or {}).get("compute", {}) if isinstance(pack.cfg, dict) else {}

    def _run(
        p0: Array2,
        ctx: dict,
        energy_fn,
        iters_override: int | None = None,
        on_iter=None,
    ):
        mode = (((comp.get("solver") or {}).get("public") or {}).get("mode", "lbfgsb"))
        tuning = ((comp.get("solver") or {}).get("tuning") or {})
        mode_cfg = dict(tuning.get(mode) or {})
        if iters_override is not None:
            iters = iters_override
        else:
            if mode in {"lbfgsb", "lbfgs"}:
                iters = mode_cfg.get("lbfgs_maxiter")
            elif mode == "semi_newton":
                iters = mode_cfg.get("sn_max_outer")
            elif mode == "hybrid":
                sn1 = mode_cfg.get("sn_max_outer", 5)
                lb = mode_cfg.get("lbfgs_maxiter", 400)
                sn2 = mode_cfg.get("sn_post_max_outer", sn1)
                iters = sn1 + lb + sn2
            else:
                iters = mode_cfg.get("maxiter")
            if iters in (None, 0):
                iters = 400
        # propagate iteration count into params for solvers that expect it
        if mode in {"lbfgsb", "lbfgs"}:
            mode_cfg.setdefault("lbfgs_maxiter", int(iters))
        elif mode == "semi_newton":
            mode_cfg.setdefault("sn_max_outer", int(iters))
        elif mode == "hybrid":
            sn1 = int(mode_cfg.get("sn_max_outer", 5))
            lb = int(mode_cfg.get("lbfgs_maxiter", 400))
            sn2 = int(mode_cfg.get("sn_post_max_outer", sn1))
            mode_cfg.setdefault("sn_max_outer", sn1)
            mode_cfg.setdefault("lbfgs_maxiter", lb)
            mode_cfg.setdefault("sn_post_max_outer", sn2)
        eng_ctx = _EngineCtx(
            labels=ctx["labels"],
            scene=ctx["scene"],
            active=ctx["active_ids"],
            cfg=pack.cfg,
            iters=int(iters),
            mode=mode,
            params=mode_cfg,
        )
        logger.debug("begin run_iters: mode=%s iters=%s", mode, iters)
        p_new, reports = _run_iters(p0, eng_ctx, energy_fn, report=True, on_iter=on_iter)
        logger.debug("end run_iters: reports=%s", len(reports))
        E, G, comps = energy_fn(
            p_new, ctx["labels"], ctx["scene"], ctx["active_ids"], comp
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
    logger.debug("begin solve with %d actions", len(actions))

    compute_cfg = pack.cfg.get("compute", {}) if isinstance(pack.cfg, dict) else {}
    pipeline_cfg = (compute_cfg.get("passes") or {}).get("pipeline")
    pm = kwargs.get("pass_manager") or kwargs.get("pm")
    if pm is None:
        pm = PassManager(compute_cfg, getattr(pack, "passes", None), pipeline=pipeline_cfg)

    pm.ensure_pass("action", position=0)
    pm.remove_pass("behavior")

    P_curr, labels, active = _init_state(pack)
    scene = pack.scene0
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

    scene_dict = scene.model_dump() if hasattr(scene, "model_dump") else scene
    sources = make_sources_from_scene(scene_dict)

    defaults = {"compute": pack.cfg.get("compute", {})}
    aux: Dict[str, Any] = {"uid": pack.uid, "rng_seed": pack.rng_seed}
    recorder.start_run(N, labels_meta, WH, sources, defaults, aux)

    tuning = (
        ((getattr(pack, "cfg", {}) or {}).get("compute", {}) or {})
        .get("solver", {})
        .get("tuning", {})
        or {}
    )
    solver_pub = ((compute_cfg.get("solver", {}) or {}).get("public", {}) or {})
    use_warmup = bool(solver_pub.get("use_warmup", True))

    capture_cfg = get_pass_cfg(
        (pm.cfg if isinstance(pm.cfg, dict) else {}),
        "capture",
        {"every": 1, "limit": None, "final_always": True, "mode": "both"},
    )
    capture_mode = capture_cfg.get("mode", "both")
    if capture_mode not in {"none", "warmup", "main", "both"}:
        logger.warning("Unknown capture.mode %s; defaulting to 'both'", capture_mode)
        capture_mode = "both"
    record_warmup = capture_mode in ("warmup", "both")
    record_main = capture_mode in ("main", "both")
    if not record_warmup and not record_main:
        logger.info("capture.mode=none: per-iteration frames will not be recorded.")
    cap_every = max(1, int(capture_cfg.get("every", 1)))
    cap_limit = capture_cfg.get("limit")
    cap_final = bool(capture_cfg.get("final_always", True))

    t_global = 0
    P_prev_full = P_curr.copy()
    if P_prev_full.ndim == 2:
        P_prev_full[~active] = np.nan
    comps_prev_full: Dict[str, np.ndarray] = {}
    for pass_id, action in enumerate(actions):
        pass_name = getattr(action, "type", None) or getattr(action, "pass_name", None) or "action"
        logger.debug("begin run_iters for action %s", pass_name)
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

        # update possibly mutated labels/active mask/position
        labels = ctx.get("labels", labels)
        active = np.asarray(ctx.get("active_ids", active), bool)
        P_curr = np.asarray(ctx.get("P", P_curr), float)
        if P_prev_full.ndim == 2:
            P_prev_full[~active] = np.nan
        active_idx = np.flatnonzero(active)

        E_pre, G_pre, comps_pre = energy_fn(P_curr, labels, scene, active, compute_cfg)
        events_pre = pm.pop_events()
        cap_cfg2 = (
            (pack.cfg.get("compute", {}).get("capture", {}) if hasattr(pack, "cfg") else {})
            or {}
        )
        pre_capture_enabled = bool(cap_cfg2.get("pre_solver_capture", True))
        if pre_capture_enabled:
            if cap_limit is None or len(recorder.frames) < cap_limit:
                P_full_pre = expand_subset(P_prev_full, active_idx, P_curr[active_idx])
                anchors_full_pre = np.asarray(
                    [label_anchor_xy(labels[j]) for j in range(N)],
                    float,
                )
                gnorm_pre = float(np.linalg.norm(G_pre)) if G_pre.size else 0.0
                ginf_pre = float(np.max(np.abs(G_pre))) if G_pre.size else 0.0
                meta_pre: Dict[str, Any] = {
                    "schema_version": "compute-v2",
                    "status": "ok",
                    "pass_name": "pre",
                    "frame_in_pass": "pre",
                    "pass_id": pass_id,
                    "events": events_pre,
                }
                metrics_pre = {
                    "E": float(E_pre),
                    "gnorm": gnorm_pre,
                    "g_inf": ginf_pre,
                }
                recorder.record_frame(
                    t=t_global,
                    P_full=P_full_pre,
                    comps_full=comps_pre,
                    E=float(E_pre),
                    active_mask=active.copy(),
                    anchors=anchors_full_pre,
                    meta_base=meta_pre,
                    metrics=metrics_pre,
                    field=None,
                    G_snapshot=G_pre,
                )
                t_global += 1
                P_prev_full = P_full_pre
                comps_prev_full = comps_pre
                logger.debug("record pre-solver frame pass_id=%d", pass_id)

        # sync recorder with possibly mutated labels
        labels_meta = [
            lbl.model_dump() if hasattr(lbl, "model_dump") else dict(lbl)
            for lbl in labels
        ]
        recorder.labels = labels_meta
        wh_rows = [getattr(lbl, "WH", None) for lbl in labels]
        if any(w is not None for w in wh_rows):
            if recorder.WH is None:
                recorder.WH = np.zeros((N, 2), float)
            for i, w in enumerate(wh_rows):
                if w is not None:
                    recorder.WH[i] = w
        else:
            recorder.WH = None

        _last_iter_recorded = -1

        def _on_iter(it: int, P_iter: np.ndarray, meta: Dict[str, Any]) -> None:
            nonlocal t_global, P_prev_full, comps_prev_full, _last_iter_recorded
            if cap_limit is not None and isinstance(cap_limit, int):
                if len(recorder.frames) >= cap_limit:
                    return
            if it % cap_every != 0:
                return
            P_full_i = expand_subset(P_prev_full, active_idx, P_iter[active_idx])
            sub_comps = meta.get("comps", {})
            if not comps_prev_full and sub_comps:
                comps_prev_full = {k: np.zeros((N, 2), float) for k in sub_comps.keys()}
            comps_full_i = expand_comps_subset(comps_prev_full, active_idx, sub_comps)
            anchors_full_i = np.asarray(
                [label_anchor_xy(labels[j]) for j in range(N)],
                float,
            )
            g_vec = meta.get("G")
            gnorm_val = float(np.linalg.norm(g_vec)) if g_vec is not None and g_vec.size else 0.0
            ginf_val = float(np.max(np.abs(g_vec))) if g_vec is not None and g_vec.size else 0.0
            algo_name = meta.get("mode") or pass_name
            meta_i: Dict[str, Any] = {
                "schema_version": "compute-v2",
                "status": "ok",
                "events": pm.pop_events(),
                "optimizer_step": {
                    "algo": algo_name,
                    "iter_in_algo": it,
                    "step_size": meta.get("step_size"),
                    "ls_evals": meta.get("ls_evals"),
                    "wolfe": meta.get("wolfe"),
                    "delta_E": meta.get("delta_E"),
                    "gnorm": gnorm_val if g_vec is not None else None,
                    "g_inf": ginf_val if g_vec is not None else None,
                },
            }
            # ensure viz contract: pass_id, pass_name, frame_in_pass
            meta_i.setdefault("pass_id", pass_id)  # viz contract
            meta_i.setdefault("pass_name", algo_name)  # viz contract
            meta_i.setdefault("frame_in_pass", it)  # iteration index for viz
            metrics_i = {"E": float(meta.get("E", 0.0)), "gnorm": gnorm_val, "g_inf": ginf_val}
            recorder.record_frame(
                t=t_global,
                P_full=P_full_i,
                comps_full=comps_full_i,
                E=meta.get("E", 0.0),
                active_mask=active.copy(),
                anchors=anchors_full_i,
                meta_base=meta_i,
                metrics=metrics_i,
                field=None,
                G_snapshot=meta.get("G"),
            )
            P_prev_full = P_full_i
            comps_prev_full = comps_full_i
            _last_iter_recorded = it
            t_global += 1

        metrics: Dict[str, Any] = {}
        if use_warmup:
            warmup_steps = int(((tuning.get("warmup", {}) or {}).get("steps", 10)) or 10)
            P_warm, metrics = run_iters(
                P_curr,
                ctx,
                energy_fn,
                iters_override=warmup_steps,
                on_iter=_on_iter if record_warmup else None,
            )
            P_curr = step_fn(P_curr, P_warm, metrics)

        P_prop, metrics = run_iters(
            P_curr,
            ctx,
            energy_fn,
            iters_override=None,
            on_iter=_on_iter if record_main else None,
        )
        P_curr = step_fn(P_curr, P_prop, metrics)

        reports = metrics.pop("reports", [])
        event = None
        m_extra: Dict[str, float] = {}
        if reports:
            event, m_extra = report_to_event(reports[-1], metrics.get("mode", ""))

        metrics_all = {"E": float(metrics.get("E", 0.0)), "gnorm": float(metrics.get("gnorm", 0.0))}
        metrics_all.update({k: float(v) for k, v in m_extra.items()})
        g_vec_final = metrics.get("G")
        if (
            g_vec_final is not None
            and getattr(g_vec_final, "size", 0)
            and "g_inf" not in metrics_all
        ):
            metrics_all["g_inf"] = float(np.max(np.abs(g_vec_final)))

        meta_base: Dict[str, Any] = {
            "schema_version": "compute-v2",
            "status": "ok",
            "reason": "",
            "timings_ms": {},
            "events": pm.pop_events(),
        }
        # ensure viz contract: pass_id, pass_name, frame_in_pass
        algo_name = metrics.get("mode", pass_name)
        meta_base.setdefault("pass_id", pass_id)  # viz contract
        meta_base.setdefault("pass_name", algo_name)  # viz contract
        meta_base.setdefault("frame_in_pass", "final")  # final marker by default for viz
        if event is not None:
            meta_base["optimizer_step"] = event

        P_full = expand_subset(P_prev_full, active_idx, P_curr[active_idx])
        sub_comps_final = metrics.get("comps", {})
        if not comps_prev_full and sub_comps_final:
            comps_prev_full = {k: np.zeros((N, 2), float) for k in sub_comps_final.keys()}
        comps_full = expand_comps_subset(comps_prev_full, active_idx, sub_comps_final)
        anchors_full = np.asarray(
            [label_anchor_xy(labels[j]) for j in range(N)],
            float,
        )

        iters = int(metrics.get("iters", len(reports)))
        need_final = cap_final and (_last_iter_recorded != (iters - 1))
        if cap_limit is not None and isinstance(cap_limit, int):
            if len(recorder.frames) >= cap_limit:
                need_final = False
        if need_final:
            metrics_all_final = metrics_all
            meta_final = {**meta_base, "frame_in_pass": "final"}  # enforce string marker for viz
            recorder.record_frame(
                t=t_global,
                P_full=P_full,
                comps_full=comps_full,
                E=metrics.get("E", 0.0),
                active_mask=active.copy(),
                 anchors=anchors_full,
                meta_base=meta_final,
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

