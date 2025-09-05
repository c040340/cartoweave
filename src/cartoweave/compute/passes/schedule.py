# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from .base import Stage, Context, ComputePass


class SchedulePass(ComputePass):
    """
    读取 SolvePack.schedule，生成阶段序列。
    每个阶段项可含：
      - iters: int               # 传给 legacy solver 的 max_iter 覆盖
      - mask_override: np.ndarray(bool[L])  # 与全局 active_mask0 取 AND
      - params: dict             # 覆盖/增量合并至 pack.params
    若 pack.schedule 为空，则生成单一阶段：使用 pack.params 与 active_mask0。
    """

    def plan_stages(self, ctx: Context) -> List[Stage]:
        pack = ctx.pack
        L = pack.L
        base_mask = pack.active_mask0
        base_params = dict(pack.params or {})
        stages_cfg: List[Dict[str, Any]] = pack.schedule or [{}]

        stages: List[Stage] = []
        for i, item in enumerate(stages_cfg):
            iters = item.get("iters", None)
            mask_over = item.get("mask_override", None)
            if mask_over is not None:
                mask = np.asarray(mask_over, dtype=bool)
                assert mask.shape == (L,), "mask_override shape must be (L,)"
                mask = np.logical_and(base_mask, mask)
            else:
                mask = base_mask

            p = dict(base_params)
            # 支持 item["params"] 或直接把其它键并入（除了 iters/mask_override）
            extra = item.get("params", {})
            if isinstance(extra, dict):
                p.update(extra)
            for k, v in item.items():
                if k not in ("iters", "mask_override", "params", "solver"):
                    p[k] = v  # 容忍直接写参数在顶层

            # 若指定 iters，则覆盖 max_iter（旧求解器通常识别这个键）
            if iters is not None:
                p = dict(p)
                p["max_iter"] = int(iters)
            solver_name = item.get("solver", None)

            stages.append(Stage(index=i, params=p, mask=mask, iters=iters, solver=solver_name))
        return stages

    # 本 pass 不负责采样
    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool:
        return True
