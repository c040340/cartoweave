from __future__ import annotations

from typing import Any

from cartoweave.utils.logging import logger
from .base import ComputePass
from . import REGISTRY, get_pass_cfg


class ActionPass(ComputePass):
    """Apply the k-th action to labels and active mask."""

    name = "action"

    def run(self, ctx: dict[str, Any]) -> None:  # pragma: no cover - thin glue
        pm = getattr(self, "pm", None)
        cfg = getattr(pm, "cfg", {}) if pm else {}
        conf = get_pass_cfg(cfg, "action", {"enable": True})
        if not conf.get("enable", True):
            return

        pack = ctx.get("pack")
        k = ctx.get("step_index", 0)
        actions = getattr(pack, "actions", [])
        if k >= len(actions):
            return
        act = actions[k]
        labels = ctx.get("labels")
        active = ctx.get("active_ids")
        lid = getattr(act, "label_id", getattr(act, "id"))
        lbl = labels[lid]
        w_curr = getattr(lbl, "WH", None)
        WH_from = tuple(w_curr) if w_curr is not None else None
        kind_from = None
        meta_lbl = getattr(lbl, "meta", None)
        if isinstance(meta_lbl, dict):
            kind_from = meta_lbl.get("mode")
        if act.type == "appear":
            active[lid] = True
            if act.WH_to is not None:
                lbl.WH = tuple(act.WH_to)
            if act.kind_to is not None:
                meta = dict(getattr(lbl, "meta", {}) or {})
                meta["mode"] = act.kind_to
                lbl.meta = meta
        elif act.type == "mutate":
            if act.WH_to is not None:
                lbl.WH = tuple(act.WH_to)
            if act.kind_to is not None:
                meta = dict(getattr(lbl, "meta", {}) or {})
                meta["mode"] = act.kind_to
                lbl.meta = meta
        elif act.type == "disappear":
            active[lid] = False

        if pm is not None:
            pm.emit_event(
                {
                    "pass": "action",
                    "info": act.type,
                    "label_id": int(lid),
                    "WH_from": WH_from,
                    "WH_to": (
                        act.WH_to.tolist()
                        if hasattr(act, "WH_to") and hasattr(act.WH_to, "tolist")
                        else getattr(act, "WH_to", None)
                    ),
                    "kind_from": kind_from,
                    "kind_to": getattr(act, "kind_to", None),
                    "global_iter": getattr(pm, "eval_index", 0),
                }
            )
        logger.debug("[action] %s label_id=%d", act.type, int(lid))


REGISTRY["action"] = (ActionPass, {})

__all__ = ["ActionPass"]
