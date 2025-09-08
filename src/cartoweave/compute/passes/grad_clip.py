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

    def __init__(self, max_norm: float | None = None, max_abs: float | None = None):
        self.max_norm = max_norm
        self.max_abs = max_abs
        self.stats: Dict[str, Any] = {"clipped_frames": 0, "max_scale_down": 0.0}

    def wrap_energy(self, energy_fn):
        """Scale ``G`` and all component forces if thresholds are exceeded."""

        stats = self.stats

        def _wrapped(P, labels, scene, active_mask, cfg):
            conf = get_pass_cfg(
                cfg,
                "grad_clip",
                {"enable": False, "max_norm": self.max_norm, "max_abs": self.max_abs},
            )
            enable = bool(conf.get("enable", False))
            mn = conf.get("max_norm")
            if mn is None and "norm_max" in conf:
                mn = conf.get("norm_max")
            ma = conf.get("max_abs")
            eps = get_eps(cfg)

            E, G, comps = energy_fn(P, labels, scene, active_mask, cfg)
            comps = dict(comps or {})

            if not enable:
                return E, G if G is not None else np.zeros_like(P), comps

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

            return E, G, comps

        return _wrapped

