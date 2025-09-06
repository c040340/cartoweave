# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .base import ComputePass
from . import get_pass_cfg


class NaNGuardPass(ComputePass):
    """Sanitize non-finite values produced by the energy function.

    Any ``NaN`` or ``Inf`` in ``E``, ``G`` or component forces is replaced
    with zero. ``E`` falls back to ``e_fallback``. Counts of fixed frames are
    recorded in ``stats``.
    """

    def __init__(self, on_nan: str = "zero", on_inf: str = "clip"):
        self.on_nan = on_nan
        self.on_inf = on_inf
        self.stats: Dict[str, Any] = {"nan_frames": 0, "inf_frames": 0, "fixed_frames": 0}

    def wrap_energy(self, energy_fn):
        """Check the output of ``energy_fn`` and zero-out non-finite values."""

        stats = self.stats

        def _wrapped(P, scene, active_mask, cfg):
            conf = get_pass_cfg(cfg, "nan_guard", {"e_fallback": 0.0, "on_nan": self.on_nan, "on_inf": self.on_inf})
            ef = float(conf.get("e_fallback", 0.0))
            E, G, comps, meta = energy_fn(P, scene, active_mask, cfg)
            meta = dict(meta or {})
            hit_nan = False
            hit_inf = False

            # Energy
            if not np.isfinite(E):
                E = ef
                hit_nan = True

            # Gradient
            if G is None:
                G = np.zeros_like(P)
            else:
                bad = ~np.isfinite(G)
                if bad.any():
                    G = G.copy()
                    G[bad] = 0.0
                    hit_nan = True
                    hit_inf = hit_inf or np.isinf(G[bad]).any()

            # Components
            comps2 = {}
            for k, V in (comps or {}).items():
                V = np.asarray(V)
                bad = ~np.isfinite(V)
                if bad.any():
                    V = V.copy()
                    V[bad] = 0.0
                    hit_nan = True
                    hit_inf = hit_inf or np.isinf(V[bad]).any()
                comps2[k] = V

            if hit_nan or hit_inf:
                stats["fixed_frames"] += 1
                if hit_nan:
                    stats["nan_frames"] += 1
                if hit_inf:
                    stats["inf_frames"] += 1

            return E, G, comps2, meta

        return _wrapped

