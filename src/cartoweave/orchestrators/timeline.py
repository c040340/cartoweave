# src/cartoweave/orchestrators/timeline.py
from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import copy
import numpy as np
from cartoweave.api import solve_frame
from ..utils.logging import logger
from cartoweave.config.layering import validate_cfg, snapshot
from cartoweave.engine import calibration as calib

Step = Dict[str, Any]
Schedule = List[Step]

def _apply_overrides(base: Dict[str, Any], step: Step) -> Dict[str, Any]:
    """返回一个新的 cfg：先复制 base，再按 step 的 scale/override 应用."""
    cfg = dict(base)  # 浅拷贝即可（值都是标量/ndarray）
    # 1) scale: { "ll.k.repulse": 0.5, "anchor.k.spring": 0.0, ... }
    for k, s in step.get("scale", {}).items():
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = float(cfg[k]) * float(s)
    # 2) override: 直接覆盖某些键
    for k, v in step.get("override", {}).items():
        cfg[k] = v
    return cfg

def _legacy_run_timeline(
    scene: Dict[str, Any],
    cfg: Dict[str, Any],
    schedule: Optional[Schedule] = None,
    *,
    mode: str = "hybrid",
    carry_P: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    执行时间线（多阶段）编排：
    - scene/cfg: 原始输入（不会原地修改）
    - schedule: 形如 [{"name": "...", "scale": {...}, "override": {...}}, ...]
      若为 None，使用一个合理默认（pre_anchor 先不拉锚，随后正常求解）
    - mode: 透传给 solve_frame（默认走现有 hybrid：半牛→LBFGS）
    - carry_P: 每步把上一阶段的解作为下一阶段 labels_init
    返回：(P_final, info)，其中 info["timeline"] 记录每步的信息
    """
    sc = copy.deepcopy(scene)
    cfg_base = dict(cfg)
    # Apply shape profile once to the base configuration so every step starts
    # from the profiled parameters.  The helper tracks whether a profile has
    # already been applied via ``_shape_applied``.
    if cfg_base.get("calib.shape.enable", False):
        calib.apply_shape_profile(cfg_base, logger)

    if schedule is None:
        schedule = [
            {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
            {"name": "main_solve"},
        ]

    logger.info("timeline start %d steps mode=%s", len(schedule), mode)
    timeline = []
    P = np.asarray(sc.get("labels_init", np.zeros((0, 2), float)), float).copy()

    k_state: Dict[str, float] = {}
    C_prev: float | None = None

    for step in schedule:
        step_name = step.get("name", "step")
        cfg_step = _apply_overrides(cfg_base, step)
        # Re-apply shape profile when the name changes between steps
        if cfg_step.get("calib.shape.name") != cfg_base.get("calib.shape.name"):
            cfg_step.pop("_shape_applied", None)
        if cfg_step.get("calib.shape.enable", False):
            calib.apply_shape_profile(cfg_step, logger)

        logger.info("timeline step '%s' start", step_name)

        if carry_P:
            sc["labels_init"] = P  # 用上一步的解做初始值

        P0 = np.asarray(sc.get("labels_init", np.zeros((0, 2), float)), float)
        C = calib.crowding_score(sc, P0, cfg_step)
        if calib.should_recalibrate_k(C, C_prev, {}, step, cfg_step):
            k_hat = calib.auto_calibrate_k(sc, P0, cfg_step, logger)
            k_state = calib.ema_update_k(
                k_state, k_hat, cfg_step.get("calib.k.ema_alpha", 0.3)
            )

        for kname, v in k_state.items():
            if kname in cfg_step:
                cfg_step[kname] = v

        P_step, info_step = solve_frame(sc, cfg_step, mode=mode)

        # 记录关键信息（可按需扩展）
        timeline.append({
            "name": step_name,
            "n_labels": int(P_step.shape[0]),
            "stage_info": info_step,  # 里面含 stage1/2 L-BFGS 的迭代统计
            "cfg_diff": step,  # 记录本阶段与 base 的差异
        })
        P = P_step
        C_prev = C
        logger.info("timeline step '%s' done", step_name)

    logger.info("timeline done")
    return P, {"timeline": timeline}


def _run_timeline_new(
    schedule: List[Dict[str, Any]],
    base_cfg: Dict[str, Any],
    solver_fn: Callable[[Dict[str, Any], Dict[str, Any]], tuple],
) -> List[tuple[np.ndarray, Dict[str, Any]]]:
    """Simplified timeline used by tests with optional calibration hooks."""
    cfg = dict(base_cfg)
    validate_cfg(cfg, phase="load")
    snapshot(cfg, "_snapshot_load")

    k_state: Dict[str, float] = {}
    C_prev: float | None = None
    results: List[tuple[np.ndarray, Dict[str, Any]]] = []

    for step_idx, step in enumerate(schedule):
        cfg_step = dict(cfg)
        cfg_step.update(step.get("overrides", {}))

        validate_cfg(cfg_step, phase="action_begin")
        snapshot(cfg_step, "_snapshot_action")

        patched = 0
        if cfg_step.get("calib.shape.enable", False):
            patched = calib.apply_shape_profile_from_cfg(cfg_step)
            if patched:
                logger.info(
                    f"[shape] applied profile={cfg_step.get('calib.shape.name')} patched={patched}"
                )

        C = calib.crowding_score(step.get("scene", {}), step.get("P0", None), cfg_step)
        if cfg_step.get("calib.k.enable", False):
            if calib.should_recalibrate_k(C, C_prev, None, step, cfg_step):
                k_hat = calib.auto_calibrate_k(
                    step.get("scene", {}), step.get("P0", None), cfg_step
                )
                if k_hat:
                    k_state = calib.ema_update_k(
                        k_state, k_hat, cfg_step.get("calib.k.ema_alpha", 0.3)
                    )
                    for k, v in k_state.items():
                        if k in cfg_step:
                            cfg_step[k] = v
                    logger.info(
                        f"[timeline] action={step_idx} k-recalibrated keys={list(k_hat.keys())}"
                    )

        P_out, info = solver_fn(step.get("scene", {}), cfg_step)
        results.append((P_out, info))

        C_prev = C
        cfg = cfg_step

    return results


def run_timeline(*args, **kwargs):
    """Dispatch to the new or legacy timeline orchestrator."""
    if args and isinstance(args[0], list):
        schedule, base_cfg, solver_fn = args[0], args[1], args[2]
        return _run_timeline_new(schedule, base_cfg, solver_fn)
    return _legacy_run_timeline(*args, **kwargs)
