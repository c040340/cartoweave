# src/cartoweave/orchestrators/timeline.py
from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional
import copy
import numpy as np
from cartoweave.api import solve_frame
from ..utils.logging import logger
from ..engine.calibration import (
    apply_shape_profile,
    auto_calibrate_k,
    crowding_score,
    should_recalibrate_k,
    ema_update_k,
)

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

def run_timeline(
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
    apply_shape_profile(cfg_base, logger)

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
        apply_shape_profile(cfg_step, logger)

        logger.info("timeline step '%s' start", step_name)

        if carry_P:
            sc["labels_init"] = P  # 用上一步的解做初始值

        P0 = np.asarray(sc.get("labels_init", np.zeros((0, 2), float)), float)
        C = crowding_score(sc, P0, cfg_step)
        if should_recalibrate_k(C, C_prev, {}, step, cfg_step):
            k_hat = auto_calibrate_k(sc, P0, cfg_step, logger)
            k_state = ema_update_k(
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
