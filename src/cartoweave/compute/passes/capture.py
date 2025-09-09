# -*- coding: utf-8 -*-
from __future__ import annotations

from cartoweave.utils.logging import logger
from .base import Context, ComputePass
from . import get_pass_cfg


class CapturePass(ComputePass):
    """Decide whether to record a frame.

    Parameters mirror those used by :class:`SolvePack.capture`.
    """

    def __init__(self, every: int = 1, final_always: bool = True,
                 limit: int | None = None, **_):
        """Store default capture cadence.

        The pass registry may provide ``every``, ``final_always`` or
        ``limit`` arguments when instantiating this pass. Runtime
        configuration can override these values via ``ctx.pack.cfg``.
        Unknown keyword arguments are ignored to keep construction
        forward-compatible.
        """
        self.every = max(1, int(every))
        self.final_always = bool(final_always)
        self.limit = limit
        self.stats = {"frames_captured": 0}

    def want_capture(self, ctx: Context, eval_index: int, frames_len: int) -> bool:
        cfg = getattr(ctx.pack, "cfg", {})
        conf = get_pass_cfg(
            cfg,
            "capture",
            {"every": self.every, "final_always": self.final_always, "limit": self.limit},
        )
        every = max(1, int(conf.get("every", 1)))
        final_always = bool(conf.get("final_always", True))
        limit = conf.get("limit", None)

        # Persist resolved values for final-frame logic and introspection.
        self.every = every
        self.final_always = final_always
        self.limit = limit

        if limit is not None and isinstance(limit, int) and frames_len >= limit:
            return False
        if every <= 1:
            should = True
        else:
            should = (eval_index % every) == 0
        if should:
            self.stats["frames_captured"] += 1
            pm = getattr(self, "pm", None)
            if pm is not None:
                pm.emit_event(
                    {
                        "pass": "capture",
                        "info": "capture",
                        "eval_index": int(eval_index),
                    }
                )
            logger.debug("[capture] capture eval=%d", int(eval_index))
        return should
