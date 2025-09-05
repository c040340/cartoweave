# -*- coding: utf-8 -*-
from __future__ import annotations
from .base import Context, ComputePass
from . import get_pass_cfg


class CapturePass(ComputePass):
    """Decide whether to record a frame.

    Parameters mirror those used by :class:`SolvePack.capture`.
    """

    def __init__(self):
        self.final_always = True
        self.stats = {"frames_captured": 0}

    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool:
        cfg = getattr(ctx.pack, "cfg", {})
        conf = get_pass_cfg(cfg, "capture", {"every": 1, "final_always": True, "limit": None})
        every = max(1, int(conf.get("every", 1)))
        limit = conf.get("limit", None)
        self.final_always = bool(conf.get("final_always", True))
        if limit is not None and isinstance(limit, int) and frames_len >= limit:
            return False
        if every <= 1:
            return True
        return (eval_index % every) == 0
