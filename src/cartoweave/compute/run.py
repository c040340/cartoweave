"""Solve function coordinating passes and compute solvers."""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import time, platform, sys

from cartoweave.contracts.solvepack import SolvePack
from cartoweave.contracts.viewpack import ViewPack, Frame
from .eval import energy_and_grad_full
from .types import _grad_metrics, Array2
from cartoweave.compute.solver.solve_layer import run_iters, SolveContext
from cartoweave.compute.passes.behavior_pass import (
    RuntimeState,
    copy_label,
    apply_behavior_step,
)
from .passes.manager import PassManager
from .passes.base import Context
from cartoweave.config.bridge import translate_legacy_keys
from .forces._common import get_eps
from cartoweave.version import __version__ as _cw_ver
from cartoweave.logging_util import get_logger


def solve_once(*, P, active, labels, scene, solver: str, params: Dict[str, Any]):
    """Single-iteration solver entry point.

    This is a thin wrapper around the project's existing solver.  For now it
    returns ``P`` unchanged; plug in ``run_iters`` or other routines as needed.
    """

    return np.asarray(P, float)


def solve_behaviors(pack: SolvePack) -> RuntimeState:
    """Apply behaviors sequentially then run solver iterations for each."""
    P = np.asarray(pack.P0, float).copy()
    active = np.asarray(pack.active0, bool).copy()
    labels = [copy_label(l) for l in pack.labels0]
    state = RuntimeState(P=P, active=active, labels=labels)

    for beh in pack.cfg.get("behaviors", []):
        state = apply_behavior_step(pack, state, beh)

        if isinstance(beh, dict):
            solver_name = beh.get("solver", "lbfgs")
            iters = int(beh.get("iters", 1))
            params = dict(beh.get("params", {}) or {})
        else:
            solver_name = getattr(beh, "solver", "lbfgs")
            iters = int(getattr(beh, "iters", 1))
            params = dict(getattr(beh, "params", {}) or {})

        for _ in range(max(1, iters)):
            state.P = solve_once(
                P=state.P,
                active=state.active,
                labels=state.labels,
                scene=pack.scene0,
                solver=solver_name,
                params=params,
            )

    return state


def solve(pack: SolvePack) -> ViewPack:
    """Run the compute pipeline and return a :class:`ViewPack`.

    The workflow is:

    ``SolvePack → passes.wrap_energy → compute solver → recorder → ViewPack``

    The function orchestrates passes that may wrap the energy function,
    mutate per-stage parameters, and decide whether each evaluation should be
    captured. Solvers are provided by :mod:`cartoweave.compute.optim` and
    dispatched via :func:`run_solver`.

    The returned summary includes:

    - ``evals`` / ``frames_captured`` – total evaluations and captured frames
    - ``stage_solvers`` – solver used for each stage
    - ``pass_stats`` – statistics reported by passes (when present)
    - ``E0`` / ``E_last`` / ``g_inf_last`` / ``moved_ratio``

    Each :class:`Frame` contains gradient norms (``g_inf``, ``g_norm``) and, if
    available, step metrics such as ``step_norm``, ``dP_inf`` and ``dE``.
    """
    pack.validate()
    cfg = translate_legacy_keys(pack.cfg)
    pack.cfg = cfg
    logger = get_logger("cartoweave.compute.run", cfg)
    t0 = time.perf_counter()
    L = pack.L
    mode = "lbfgs" if pack.mode is None else str(pack.mode).lower()

    # PassManager 构造与阶段规划
    pm = PassManager(cfg, pack.passes)
    passes = pm.passes
    capture_pass = next((p for p in passes if hasattr(p, "final_always")), None)
    ctx = Context(pack=pack, stages=[], eval_index=0, stage_index=0)
    stages = pm.plan_stages(ctx, pack.stages)
    ctx.stages = stages

    # 包裹能量和步长
    base_energy = pack.energy_and_grad or energy_and_grad_full
    # Allow legacy energy functions with signature (P, scene, active, cfg)
    if base_energy is not energy_and_grad_full:
        import inspect

        try:
            if len(inspect.signature(base_energy).parameters) == 4:
                orig_energy = base_energy

                def _legacy_energy(P, labels, scene, active, cfg):
                    return orig_energy(P, scene, active, cfg)

                base_energy = _legacy_energy
        except Exception:
            pass

    def _energy_with_meta(P, labels, scene, active, cfg):
        out = base_energy(P, labels, scene, active, cfg)
        if isinstance(out, tuple) and len(out) == 3:
            E, G, comps = out
            return E, G, comps, {}
        return out

    energy_fn = pm.wrap_energy(_energy_with_meta)
    apply_step = pm.wrap_step(lambda P_old, P_prop, metrics: P_prop)
    logger.info("solve.start", extra={"extra": {"stages": len(stages)}})

    frames: List[Frame] = []
    terms_used_set = set()
    stage_solver_names: List[str] = []
    last_P: Array2 | None = None
    last_E: float | None = None
    last_record: Dict[str, Any] | None = None
    mask_popcount: List[int] = []

    # recorder 闭包：带入 capture、stage 索引
    def make_recorder(stage_idx: int, stage_mask: np.ndarray, P_start: Array2):
        def _rec(rec: Dict[str, Any]):
            nonlocal ctx, frames, terms_used_set, last_P, last_E, last_record, P_start
            P_prop = rec.get("P", None)
            if P_prop is None:
                return
            P_prop = np.asarray(P_prop)
            if P_prop.shape != (L, 2):
                return  # 本步仍然要求全长；子集帧先跳过

            P_old = last_P if last_P is not None else P_start
            P = apply_step(P_old, P_prop, rec)

            G = rec.get("G", rec.get("grad", None))
            if G is not None:
                G = np.asarray(G)
                if G.shape != (L, 2):
                    G = None

            comps = rec.get("comps", {}) or {}
            comps_full: Dict[str, Array2] = {}
            for k, v in comps.items():
                v = np.asarray(v)
                if v.shape == (L, 2):
                    comps_full[k] = v
                    terms_used_set.add(k)

            E = float(rec.get("E", rec.get("energy", np.nan)))
            mask = rec.get("mask", stage_mask)
            mask = np.asarray(mask)
            mask = mask if mask.shape == (L,) else stage_mask

            metrics = _grad_metrics(G)

            ls_iters = rec.get("ls_iters", rec.get("line_search_iters", None))
            if ls_iters is not None:
                metrics["ls_iters"] = float(ls_iters)
            alpha = rec.get("alpha", rec.get("step_size", None))
            if alpha is not None:
                metrics["alpha"] = float(alpha)
            accept = rec.get("accept", rec.get("accepted", None))
            if accept is not None:
                metrics["accept"] = float(1.0 if accept else 0.0)

            dP = P - P_old
            metrics["step_norm"] = float(np.linalg.norm(dP))
            metrics["dP_inf"] = float(np.linalg.norm(dP, ord=np.inf))

            if last_E is not None and np.isfinite(E) and np.isfinite(last_E):
                metrics["dE"] = float(E - last_E)

            meta = rec.get("meta", {}) or {}

            frame_kwargs = dict(
                E=E,
                P=P,
                G=G if G is not None else np.zeros_like(P),
                comps=comps_full,
                mask=mask,
                metrics=metrics,
                meta=meta,
            )

            if pm.want_capture(ctx, ctx.eval_index, len(frames)):
                frames.append(Frame(i=ctx.eval_index, stage=stage_idx, **frame_kwargs))
                if capture_pass is not None:
                    capture_pass.stats["frames_captured"] = capture_pass.stats.get("frames_captured", 0) + 1

            last_record = (frame_kwargs, stage_idx, ctx.eval_index)
            last_P = P
            last_E = E
            ctx.eval_index += 1

        return _rec

    # 串行跑各阶段
    P_curr = pack.P0
    result: Dict[str, Any] = {}
    for s_idx, st in enumerate(stages):
        ctx.stage_index = s_idx
        mode_stage = (st.solver or pack.mode or "lbfgs").lower()
        stage_solver_names.append(mode_stage)
        mask_popcount.append(int(st.mask.sum()))

        rec = make_recorder(s_idx, st.mask, P_curr)
        labels_all = pack.scene0.get("labels")
        E0, G0, comps0, _ = energy_fn(P_curr, labels_all, pack.scene0, st.mask, cfg)
        rec({"P": P_curr, "G": G0, "comps": comps0, "E": E0})

        ctx_stage = SolveContext(
            labels=labels_all,
            scene=pack.scene0,
            active=st.mask,
            cfg=cfg,
            iters=int(st.iters or 0),
            mode=mode_stage,
            params=st.params or {},
        )

        def _energy_no_meta(P, labels, scene, active, cfg):
            E, G, comps, _ = energy_and_grad_full(P, labels, scene, active, cfg)
            return E, G, comps

        P_curr, _ = run_iters(P_curr, ctx_stage, _energy_no_meta, report=False)
        ctx.eval_index += ctx_stage.iters

        E1, G1, comps1, _ = energy_fn(P_curr, labels_all, pack.scene0, st.mask, cfg)
        rec({"P": P_curr, "G": G1, "comps": comps1, "E": E1})
        result = {"P": P_curr}

        # 若未记录最终一帧，则按需补齐
        last_eval_idx = ctx.eval_index - 1
        has_final = any(
            f.stage == s_idx and f.i == last_eval_idx for f in frames
        )
        need_final = (capture_pass is not None and capture_pass.final_always) or not any(
            f.stage == s_idx for f in frames
        )
        if need_final and not has_final and last_record is not None:
            frame_kwargs, stage_rec, idx_rec = last_record
            frames.append(Frame(i=idx_rec, stage=stage_rec, **frame_kwargs))

    if not frames and last_record is not None:
        frame_kwargs, stage_rec, idx_rec = last_record
        frames.append(Frame(i=idx_rec, stage=stage_rec, **frame_kwargs))

    last = frames[-1]
    terms_used = sorted(terms_used_set)
    eps = get_eps(cfg)
    iters = ctx.eval_index
    stop_reason = str(result.get("stop_reason", "unknown"))
    converged = bool(result.get("converged", stop_reason in ("gtol", "ftol")))
    summary = {
        "evals": iters,
        "frames_captured": len(frames),
        "iters": iters,
        "stages": len(stages),
        "converged": converged,
        "stop_reason": stop_reason,
        "E0": float(frames[0].E),
        "E_last": float(last.E),
        "g_inf_last": float(last.metrics.get("g_inf", np.nan)),
        "moved_ratio": float(
            np.mean(np.linalg.norm(last.P - pack.P0, axis=1) > eps)
        ),
        "stage_solvers": stage_solver_names,
        "stages_params": [getattr(st, "params", dict(getattr(st, "__dict__", {}))) for st in stages],
        "mask_popcount": mask_popcount,
        "schema_version": "v1",
        "runtime": {
            "cartoweave": _cw_ver,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    summary["time_ms"] = int((time.perf_counter() - t0) * 1000)
    summary.setdefault("pass_stats", pm.collect_stats())
    view = ViewPack(
        L=L,
        mode=mode,
        params_used=stages[-1].params if stages else (pack.params or {}),
        terms_used=terms_used,
        frames=frames,
        last=last,
        summary=summary,
    )
    logger.info(
        "solve.done",
        extra={"extra": {"frames": view.summary.get("frames_captured", 0), "time_ms": summary["time_ms"]}},
    )
    return view
