# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any
from copy import deepcopy

import numpy as np

from cartoweave.utils.logging import logger
from cartoweave.compute.eval import energy_and_grad_full
from .base import ComputePass
from . import REGISTRY, get_pass_cfg


class LabelRelaxPass(ComputePass):
    """Do multi-step gradient relaxation for newly appeared labels."""

    name = "label_relax"

    def run(self, ctx: dict[str, Any]) -> None:  # pragma: no cover - thin glue
        pm = getattr(self, "pm", None)
        cfg = getattr(pm, "cfg", {}) if pm else {}
        conf = get_pass_cfg(
            cfg, "label_relax", {"enable": True, "step_size": 0.1, "step_count": 5}
        )
        if not conf.get("enable", True):
            return

        pack = ctx.get("pack")
        step_idx = int(ctx.get("step_index", 0))
        actions = getattr(pack, "actions", []) if pack is not None else []
        if step_idx >= len(actions):
            return
        act = actions[step_idx]
        if getattr(act, "type", None) != "appear":
            return
        lbl_id = int(getattr(act, "id", -1))
        if lbl_id < 0:
            return

        P = ctx.get("P")
        labels = ctx.get("labels")
        scene = ctx.get("scene")
        active = ctx.get("active_ids")
        if P is None or labels is None or scene is None or active is None:
            return

        cfg_local = deepcopy(getattr(pack, "cfg", {})) if pack is not None else {}
        comp = cfg_local.setdefault("compute", {})
        pub = comp.setdefault("public", {})
        forces = pub.setdefault("forces", {})
        forces.setdefault("anchor", {}).setdefault("spring", {})["enable"] = False
        forces.setdefault("area", {}).setdefault("embed", {})["enable"] = False

        step_size = float(conf.get("step_size", 0.1))
        steps = int(conf.get("step_count", 5))
        done = 0
        for _ in range(max(1, steps)):
            _, g, _ = energy_and_grad_full(P, labels, scene, active, comp)
            if g is None or not np.isfinite(g[lbl_id]).all():
                break
            P[lbl_id] = P[lbl_id] - step_size * g[lbl_id]
            done += 1
        ctx["P"] = P

        if pm is not None:
            pm.emit_event(
                {
                    "pass": "label_relax",
                    "label_id": lbl_id,
                    "steps": done,
                    "global_iter": getattr(pm, "eval_index", 0),
                }
            )
        logger.info(
            "[label_relax] label_id=%d step_size=%s steps=%d", lbl_id, step_size, done
        )


REGISTRY["label_relax"] = (LabelRelaxPass, {})

__all__ = ["LabelRelaxPass"]
