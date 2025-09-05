# -*- coding: utf-8 -*-
from __future__ import annotations
from .base import Context, ComputePass


class CapturePass(ComputePass):
    """按 every / limit / final_always 控制采样。"""

    def __init__(self, every: int = 1, limit=None, final_always: bool = True):
        self.every = max(1, int(every))
        self.limit = limit
        self.final_always = bool(final_always)

    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool:
        if self.limit is not None and isinstance(self.limit, int) and frames_len >= self.limit:
            return False
        if self.every <= 1:
            return True
        return (eval_index % self.every) == 0
