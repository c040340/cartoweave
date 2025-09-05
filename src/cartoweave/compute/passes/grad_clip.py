# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .base import ComputePass


class GradClipPass(ComputePass):
    """梯度裁剪：支持范数裁剪和元素级裁剪。"""

    def __init__(self, max_norm: float | None = None, max_abs: float | None = None, eps: float = 1e-12):
        self.max_norm = float(max_norm) if max_norm is not None else None
        self.max_abs = float(max_abs) if max_abs is not None else None
        self.eps = float(eps)
        self.stats: Dict[str, Any] = {"clipped_frames": 0, "max_scale_down": 0.0}

    def wrap_energy(self, energy_fn):
        mn, ma, eps = self.max_norm, self.max_abs, self.eps
        stats = self.stats

        def _wrapped(P, scene, active_mask, cfg):
            E, G, comps, meta = energy_fn(P, scene, active_mask, cfg)
            comps = dict(comps or {})

            if G is None:
                G = np.zeros_like(P)

            # Element-wise clipping
            if ma is not None:
                np.clip(G, -ma, ma, out=G)
                for k in list(comps.keys()):
                    np.clip(comps[k], -ma, ma, out=comps[k])

            scale = 1.0
            if mn is not None:
                gnorm = float(np.linalg.norm(G))
                if gnorm > mn + eps:
                    scale = mn / (gnorm + eps)
                    G = G * scale
                    for k in list(comps.keys()):
                        comps[k] = comps[k] * scale
                    stats["clipped_frames"] += 1
                    if scale < stats["max_scale_down"] or stats["max_scale_down"] == 0.0:
                        stats["max_scale_down"] = scale

            return E, G, comps, meta

        return _wrapped

