# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from .base import Stage, Context, ComputePass


class SchedulePass(ComputePass):
    """Expand :class:`SolvePack.schedule` into concrete stages.

    Each stage item may specify:

    - ``iters`` – overrides ``max_iter`` for the engine solver
    - ``params`` – dict merged into the pack's ``params``

    When ``SolvePack.schedule`` is empty, a single stage using the
    pack-wide ``params`` and mask is produced.
    """

    def plan_stages(self, ctx: Context, stages: List[Stage]) -> List[Stage]:
        pack = ctx.pack
        L = pack.L
        base_mask = pack.active_mask0
        base_params = dict(getattr(pack, "params", {}) or {})
        stages_cfg: List[Dict[str, Any]] = getattr(pack, "stages", None) or [{}]

        stages: List[Stage] = []
        for i, item in enumerate(stages_cfg):
            iters = item.get("iters", None)
            mask = base_mask

            p = dict(base_params)
            # 支持 item["params"] 或直接把其它键并入（除了 iters）
            extra = item.get("params", {})
            if isinstance(extra, dict):
                p.update(extra)
            for k, v in item.items():
                if k not in ("iters", "params", "solver"):
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
