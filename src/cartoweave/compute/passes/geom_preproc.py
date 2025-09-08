# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from copy import deepcopy

from .base import ComputePass


class GeomPreprocPass(ComputePass):
    """Clean up line polylines before each stage.

    Removes consecutive duplicate vertices and near-zero length segments
    from line label polylines. Does not mutate the original ``SolvePack``;
    cleaned labels are cached per stage and passed to the wrapped energy
    function. When a polyline collapses to <2 vertices, the label is marked
    inert for line-based terms via its ``meta`` field.
    """

    name = "geom_preproc"

    def __init__(self, tiny_eps: float = 1e-9):
        self.tiny_eps = float(tiny_eps)
        self.stage_cache: Dict[int, List[Any]] = {}

    def _clean_polyline(self, pts: np.ndarray) -> (List[tuple[float, float]], int, int):
        collapsed = 0
        dropped = 0
        kept: List[np.ndarray] = [pts[0]]
        for p in pts[1:]:
            dist = float(np.linalg.norm(p - kept[-1]))
            if dist == 0.0:
                collapsed += 1
                continue
            if dist < self.tiny_eps:
                dropped += 1
                continue
            kept.append(p)
        cleaned = [tuple(map(float, v)) for v in kept]
        return cleaned, collapsed, dropped

    def wrap_energy(self, energy_fn):
        from . import get_pass_cfg  # local import to avoid cycles

        pm = getattr(self, "pm", None)
        cfg = getattr(pm, "cfg", {}) if pm else {}
        conf = get_pass_cfg(cfg, "geom_preproc", {"enable": True, "tiny_eps": self.tiny_eps})

        if not conf.get("enable", True):
            return energy_fn

        self.tiny_eps = float(conf.get("tiny_eps", self.tiny_eps))
        tiny = self.tiny_eps

        def _wrapped(P, labels, scene, mask, cfg):
            key = 0
            if key not in self.stage_cache:
                collapsed = 0
                dropped = 0
                affected = 0
                cleaned_labels: List[Any] = []
                for lab in labels or []:
                    if getattr(lab, "kind", None) != "line":
                        cleaned_labels.append(lab)
                        continue
                    poly = getattr(lab, "polyline", None) or []
                    arr = np.asarray(poly, float).reshape(-1, 2)
                    if arr.shape[0] >= 2:
                        new_poly, c_cnt, d_cnt = self._clean_polyline(arr)
                        collapsed += c_cnt
                        dropped += d_cnt
                    else:
                        new_poly, c_cnt, d_cnt = [], 0, 0
                    lab2 = deepcopy(lab)
                    if len(new_poly) < 2:
                        meta = dict(getattr(lab2, "meta", {}) or {})
                        meta["inert_line"] = True
                        lab2.meta = meta
                        lab2.polyline = []
                        if arr.shape[0] >= 2:
                            affected += 1
                    else:
                        if len(new_poly) != arr.shape[0]:
                            affected += 1
                        lab2.polyline = new_poly
                    cleaned_labels.append(lab2)
                self.stage_cache[key] = cleaned_labels
                if pm:
                    pm.emit_event(
                        {
                            "pass": "geom_preproc",
                            "stage_id": 0,
                            "global_iter": getattr(pm, "eval_index", 0),
                            "info": "cleanup",
                            "tiny_eps": float(tiny),
                            "metrics": {
                                "collapsed": collapsed,
                                "dropped_segments": dropped,
                                "affected_labels": affected,
                            },
                        }
                    )
            labels_use = self.stage_cache.get(key, labels)
            return energy_fn(P, labels_use, scene, mask, cfg)

        return _wrapped
