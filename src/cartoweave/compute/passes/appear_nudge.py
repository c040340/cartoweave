# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.compute.passes.base import ComputePass
from cartoweave.compute.passes import REGISTRY, get_pass_cfg
from cartoweave.utils.logging import logger

EPS = 1e-12


class AppearNudgePass(ComputePass):
    """轻微推动新激活的 label，避免锚点弹簧方向退化。"""

    name = "appear_nudge"

    def __init__(
        self,
        enable: bool = False,
        step_px: float = 1.0,
        max_px: float | None = 3.0,
        seed_offset: int = 0,
    ) -> None:
        self.enable = bool(enable)
        self.step_px = float(step_px)
        self.max_px = None if max_px is None else float(max_px)
        self.seed_offset = int(seed_offset)
        self._prev_active: np.ndarray | None = None
        self.stats = {"nudged": 0}

    def run(self, ctx: Dict[str, Any]) -> None:
        pm = getattr(self, "pm", None)
        cfg = getattr(pm, "cfg", {}) if pm else {}
        conf = get_pass_cfg(
            cfg,
            "appear_nudge",
            {
                "enable": self.enable,
                "step_px": self.step_px,
                "max_px": self.max_px,
                "seed_offset": self.seed_offset,
            },
        )
        enable = bool(conf.get("enable", self.enable))
        step_px = float(conf.get("step_px", self.step_px))
        max_px = conf.get("max_px", self.max_px)
        if max_px is not None:
            max_px = float(max_px)
        seed_offset = int(conf.get("seed_offset", self.seed_offset))

        if not enable:
            self._prev_active = (
                np.asarray(ctx.get("active_ids"), bool).copy()
                if ctx.get("active_ids") is not None
                else None
            )
            return

        P = np.asarray(ctx.get("P"), float)
        if P.ndim != 2 or P.shape[1] != 2:
            raise AssertionError("P must be (N,2) float")

        labels = ctx.get("labels")
        scene = ctx.get("scene")
        active = np.asarray(ctx.get("active_ids"), bool)
        comp_cfg = (
            (
                ctx.get("pack").cfg.get("compute")
                if ctx.get("pack") is not None
                else ctx.get("cfg", {}).get("compute", {})
            )
            or {}
        )

        if self._prev_active is None:
            self._prev_active = active.copy()
            return
        new_mask = active & (~self._prev_active)
        if not np.any(new_mask):
            self._prev_active = active.copy()
            return
        new_ids = np.flatnonzero(new_mask)

        tmp_comp = _deepcopy_forces_cfg_without_anchor_spring(comp_cfg)

        E, G, _ = energy_and_grad_full(P, labels, scene, active, tmp_comp)
        F = -np.asarray(G, float)

        moved = 0
        for i in new_ids:
            v = F[i]
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n < EPS:
                rng = np.random.default_rng(seed=int(seed_offset + i))
                v = rng.normal(size=2)
                n = float(np.linalg.norm(v))
            d = (v / max(n, EPS)) * step_px
            if max_px is not None:
                m = float(np.linalg.norm(d))
                if m > max_px and m > EPS:
                    d = d * (max_px / m)
            P[i] = P[i] + d
            moved += 1

        ctx["P"] = P
        self._prev_active = active.copy()
        self.stats["nudged"] += moved
        if moved:
            logger.debug(
                "[appear_nudge] nudged=%d, step_px=%s, max_px=%s",
                moved,
                step_px,
                max_px,
            )


def _deepcopy_forces_cfg_without_anchor_spring(comp_cfg: dict) -> dict:
    import copy

    cfg2 = copy.deepcopy(comp_cfg or {})
    forces = cfg2.setdefault("forces", {})
    anchor = forces.setdefault("anchor", {})
    spring = anchor.setdefault("spring", {})
    spring["enable"] = False
    return cfg2


REGISTRY["appear_nudge"] = (AppearNudgePass, {})

__all__ = ["AppearNudgePass"]
