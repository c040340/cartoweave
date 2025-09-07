from __future__ import annotations

from typing import Any

from .base import ComputePass
from . import REGISTRY


class ActionPass(ComputePass):
    """Apply the k-th action to labels and active mask."""

    name = "action"

    def run(self, ctx: dict[str, Any]) -> None:  # pragma: no cover - thin glue
        pack = ctx.get("pack")
        k = ctx.get("step_index", 0)
        actions = getattr(pack, "actions", [])
        if k >= len(actions):
            return
        act = actions[k]
        labels = ctx.get("labels")
        active = ctx.get("active_ids")
        lbl = labels[act.id]
        if act.type == "appear":
            active[act.id] = True
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
            active[act.id] = False


REGISTRY["action"] = (ActionPass, {})

__all__ = ["ActionPass"]
