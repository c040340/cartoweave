"""Action-driven solver entrypoint."""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from cartoweave.contracts.solvepack import SolvePack
from .eval import energy_and_grad_full
from .optim.loop import LoopContext as _EngineCtx, run_iters as _run_iters
from .passes import PassManager
from .recorder import Recorder

logger = logging.getLogger(__name__)
Array2 = np.ndarray


def _init_state(pack: SolvePack) -> tuple[Array2, list[Any], np.ndarray]:
    p = np.asarray(pack.P0, float).copy()
    labels = [deepcopy(lbl) for lbl in pack.labels0]
    active = np.asarray(pack.active0, bool)
    return p, labels, active


def _build_run_iters(pack: SolvePack):
    compute_cfg = pack.cfg.get("compute", {}) if isinstance(pack.cfg, dict) else {}

    def _run(p0: Array2, ctx: dict, energy_fn, iters_override: int | None = None):
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
        p_new, reports = _run_iters(p0, eng_ctx, energy_fn, report=True)
        E, G, comps = energy_fn(p_new, ctx["labels"], ctx["scene"], ctx["active_ids"], compute_cfg)
        gnorm = float(np.linalg.norm(G)) if G is not None and G.size else 0.0
        return p_new, {
            "E": E,
            "G": G,
            "gnorm": gnorm,
            "iters": reports[-1].it + 1 if reports else 0,
            "comps": comps,
            "mode": mode,
        }

    return _run


def solve(pack: SolvePack, *args, **kwargs):
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
    recorder: Recorder = pm.build_recorder()
    run_iters = _build_run_iters(pack)

    tuning = ((getattr(pack, "cfg", {}) or {}).get("compute", {}) or {}).get("solver", {}).get("tuning", {}) or {}
    per_action_iters = int(((tuning.get("warmup", {}) or {}).get("steps", 1)) or 1)

    for k in range(len(actions)):
        ctx = {
            "pack": pack,
            "scene": scene,
            "labels": labels,
            "active_ids": active,
            "P": P_curr,
            "step_index": k,
        }
        pm.run_step(ctx)
        P_prop, metrics = run_iters(P_curr, ctx, energy_fn, iters_override=per_action_iters)
        P_curr = step_fn(P_curr, P_prop, metrics)
        recorder.capture_step_end(k, P_curr, labels, metrics)
        recorder.record_events(pm.pop_events())

    return recorder.to_viewpack(P_curr, labels)


__all__ = ["solve"]
