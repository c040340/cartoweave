# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .base import ComputePass
from . import get_pass_cfg
from cartoweave.compute.forces._common import get_eps


class GradClipPass(ComputePass):
    """Clip gradients and component forces.

    Supports global L2-norm clipping via ``max_norm`` and element-wise
    clipping via ``max_abs``. When clipping occurs, ``stats`` records how many
    frames were affected and the strongest scale factor applied.
    """

    def __init__(self):
        self.stats: Dict[str, Any] = {"clipped_frames": 0, "max_scale_down": 0.0}

    def wrap_energy(self, energy_fn):
        """Scale ``G`` and all component forces if thresholds are exceeded."""

        stats = self.stats

        def _wrapped(P, scene, active_mask, cfg):
            conf = get_pass_cfg(cfg, "grad_clip", {"max_norm": None, "max_abs": None})
            mn = conf.get("max_norm")
            ma = conf.get("max_abs")
            eps = get_eps(cfg)

            E, G, comps, meta = energy_fn(P, scene, active_mask, cfg)
            comps = dict(comps or {})

            if G is None:
                G = np.zeros_like(P)

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

