"""Thin orchestrator wiring passes, optim loop and recorder."""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.compute.optim.loop import LoopContext as _EngineCtx
from cartoweave.compute.optim.loop import run_iters as _run_iters
from cartoweave.compute.passes.base import Context, Stage
from cartoweave.compute.passes.manager import PassManager
from cartoweave.compute.types import ViewPack
from cartoweave.contracts.solvepack import SolvePack

logger = logging.getLogger(__name__)

Array2 = np.ndarray


@dataclass
class SolveContext:
    """Minimal context passed to the optimisation loop."""

    P: Array2
    mask: np.ndarray
    cfg: dict[str, Any]
    scene: dict[str, Any]
    labels: list[Any]
    stage: Stage


def solve_behaviors(pack: SolvePack) -> tuple[Array2, list[Any]]:
    """Initialise state and labels from ``SolvePack``."""

    p = np.asarray(pack.P0, float).copy()
    labels = [deepcopy(lbl) for lbl in pack.labels0]
    return p, labels


def run_iters(
    p0: Array2,
    ctx: SolveContext,
    energy_fn: Callable[[Array2, Any, Any, np.ndarray, dict[str, Any]], tuple[float, Array2, dict[str, Array2]]],
) -> tuple[Array2, dict[str, Any]]:
    """Dispatch to optimisation loop and collect metrics."""

    mode = ctx.stage.solver or ctx.cfg.get("solver", {}).get("public", {}).get("mode", "lbfgsb")
    tuning = ctx.cfg.get("solver", {}).get("tuning", {})
    iters = ctx.stage.iters or tuning.get(mode, {}).get("maxiter", 1)
    eng_ctx = _EngineCtx(
        labels=ctx.labels,
        scene=ctx.scene,
        active=ctx.mask,
        cfg=ctx.cfg,
        iters=int(iters),
        mode=mode,
        params=getattr(ctx.stage, "params", {}) or {},
    )
    p_new, reports = _run_iters(p0, eng_ctx, energy_fn, report=True)
    e, g, comps = energy_fn(p_new, ctx.labels, ctx.scene, ctx.mask, ctx.cfg)
    gnorm = float(np.linalg.norm(g)) if g is not None and g.size else 0.0
    eps_norm = (
        ctx.cfg.get("solver", {})
        .get("internals", {})
        .get("stability", {})
        .get("eps_norm")
    )
    if eps_norm is not None:
        gnorm = float(max(gnorm, float(eps_norm)))
    return p_new, {
        "E": e,
        "G": g,
        "gnorm": gnorm,
        "iters": reports[-1].it + 1 if reports else 0,
        "comps": comps,
        "mode": mode,
    }


def solve(pack: SolvePack) -> ViewPack:
    """Entry point solving the compute pipeline."""

    p_curr, labels = solve_behaviors(pack)
    active = np.asarray(pack.active0, bool)
    cfg = pack.cfg["compute"]
    pm = PassManager(cfg, getattr(pack, "passes", None))
    ctx_pm = Context(pack=pack, stages=[], eval_index=0, stage_index=0)
    stages = pm.plan_stages(ctx_pm, getattr(pack, "stages", None))
    energy_fn = pm.wrap_energy(energy_and_grad_full)
    step_fn = pm.wrap_step(lambda p_old, p_new, metrics: p_new)
    recorder = pm.build_recorder()
    scene_dict = pack.scene0.model_dump()

    for sid, stage in enumerate(stages):
        st_mask = pm.stage_mask(active, stage)
        ctx = SolveContext(
            P=p_curr,
            mask=st_mask,
            cfg=cfg,
            scene=scene_dict,
            labels=labels,
            stage=stage,
        )
        p_prop, metrics = run_iters(p_curr, ctx, energy_fn)
        p_curr = step_fn(p_curr, p_prop, metrics)
        recorder.capture_stage_end(
            sid,
            p_curr,
            labels,
            metrics,
            st_mask,
            final=(sid == len(stages) - 1),
        )
        logger.debug(
            "stage=%d mode=%s iters=%d E=%.4g Gnorm=%.4g",
            sid,
            metrics.get("mode"),
            metrics.get("iters", 0),
            metrics.get("E", 0.0),
            metrics.get("gnorm", 0.0),
        )
        recorder.record_events(pm.pop_events())

    return recorder.to_viewpack(p_curr, labels)
