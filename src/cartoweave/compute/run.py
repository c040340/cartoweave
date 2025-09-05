"""Solve function coordinating passes and engine solvers."""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np

from .pack import SolvePack
from .eval import energy_and_grad_full as _default_energy
from .types import ViewPack, Frame, _grad_metrics, Array2
from .optim import run_via_engine_solver
from .passes import build_passes
from .passes.base import Context


def solve(pack: SolvePack) -> ViewPack:
    """Run the compute pipeline and return a :class:`ViewPack`.

    The workflow is:

    ``SolvePack → passes.wrap_energy → engine solver → recorder → ViewPack``

    The function orchestrates passes that may wrap the energy function,
    mutate per-stage parameters, and decide whether each evaluation should be
    captured. Solvers themselves live in :mod:`cartoweave.engine.solvers` and
    are called via :func:`run_via_engine_solver`.

    The returned summary includes:

    - ``evals`` / ``frames_captured`` – total evaluations and captured frames
    - ``stage_solvers`` – solver used for each stage
    - ``pass_stats`` – statistics reported by passes (when present)
    - ``E0`` / ``E_last`` / ``g_inf_last`` / ``moved_ratio``

    Each :class:`Frame` contains gradient norms (``g_inf``, ``g_norm``) and, if
    available, step metrics such as ``step_norm``, ``dP_inf`` and ``dE``.
    """
    pack.validate()
    L = pack.L
    mode = "lbfgs" if pack.mode is None else str(pack.mode).lower()

    # 构造 passes
    passes = build_passes(pack.passes, pack.capture)

    # 按顺序包裹能量函数
    energy_fn = pack.energy_and_grad or _default_energy
    for p in passes:
        energy_fn = getattr(p, "wrap_energy", lambda f: f)(energy_fn)

    capture_pass = next((p for p in passes if hasattr(p, "final_always")), None)
    ctx = Context(pack=pack, stages=[], eval_index=0, stage_index=0)
    # 由 schedule 生成阶段
    stages = passes[0].plan_stages(ctx)  # 第一个总是 schedule
    ctx.stages = stages

    # 允许 pass 对阶段参数做二次注入（step_limit 等）
    for p in passes:
        mutate = getattr(p, "mutate_stage", None)
        if mutate:
            for st in stages:
                mutate(st)

    frames: List[Frame] = []
    terms_used_set = set()
    stage_solver_names: List[str] = []
    last_P: Array2 | None = None
    last_E: float | None = None
    last_record: Dict[str, Any] | None = None

    # recorder 闭包：带入 capture、stage 索引
    def make_recorder(stage_idx: int, stage_mask: np.ndarray):
        def _rec(rec: Dict[str, Any]):
            nonlocal ctx, frames, terms_used_set, last_P, last_E, last_record
            P = rec.get("P", None)
            if P is None:
                return
            P = np.asarray(P)
            if P.shape != (L, 2):
                return  # 本步仍然要求全长；子集帧先跳过

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

            if last_P is not None:
                dP = P - last_P
                metrics["step_norm"] = float(np.linalg.norm(dP))
                metrics["dP_inf"] = float(np.linalg.norm(dP, ord=np.inf))
            else:
                metrics["step_norm"] = 0.0
                metrics["dP_inf"] = 0.0

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

            want = True
            for p in passes:
                want = p.want_capture(ctx, ctx.eval_index, len(frames)) and want
            if want:
                frames.append(Frame(i=ctx.eval_index, stage=stage_idx, **frame_kwargs))

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
        result = run_via_engine_solver(
            mode=mode_stage,
            P0=P_curr,
            scene=pack.scene,
            active_mask=st.mask,
            params=st.params or {},
            energy_and_grad=energy_fn,
            recorder=make_recorder(s_idx, st.mask),
        )
        P_final = np.asarray(result.get("P_final", P_curr))
        if P_final.shape == (L, 2):
            P_curr = P_final  # 作为下一阶段初值

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
    # 汇总
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
            np.mean(np.linalg.norm(last.P - pack.P0, axis=1) > 1e-9)
        ),
        "stage_solvers": stage_solver_names,
    }

    # 汇总各 pass 的统计信息
    pass_stats = {}
    for p in passes:
        stats = getattr(p, "stats", None)
        if stats:
            pass_stats[p.__class__.__name__] = dict(stats)
    if pass_stats:
        summary["pass_stats"] = pass_stats
    return ViewPack(
        L=L,
        mode=mode,
        params_used=stages[-1].params if stages else (pack.params or {}),
        terms_used=terms_used,
        frames=frames,
        last=last,
        summary=summary,
    )
