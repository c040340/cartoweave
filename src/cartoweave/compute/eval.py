# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Any, List
import numpy as np

from cartoweave.utils.checks import merge_sources
from cartoweave.engine.forces import REGISTRY as _ENG_REG, enabled_terms as _eng_enabled
from .forces import REGISTRY as _CMP_REG

Array2 = np.ndarray


def _as_active_ids(mask: np.ndarray) -> List[int]:
    mask = np.asarray(mask, bool)
    return np.where(mask)[0].tolist()


def _clip_if_needed(F: Array2, cfg: dict) -> Array2:
    clamp_max = (
        cfg.get("solver", {})
        .get("tuning", {})
        .get("clamp", {})
        .get("optimize_force_max")
    )
    if clamp_max is None:
        return F
    cm = float(clamp_max)
    np.clip(F, -cm, cm, out=F)
    return F


def _get_weights(cfg: dict):
    return (
        cfg.get("solver", {})
        .get("internals", {})
        .get("weights", {})
    ) or {}


def _weight_for(name: str, wmap: dict) -> float:
    if name in wmap:
        return float(wmap[name])
    for k, v in wmap.items():
        if k.endswith(".*") and name.startswith(k[:-2] + "."):
            return float(v)
        if "." not in k and name.startswith(k + "."):
            return float(v)
    return 1.0


def _energy_and_grad_full_compute(P: Array2, scene: Dict[str, Any], active_mask: np.ndarray, cfg: dict):
    """两阶段聚合所有启用项，返回 (E_total, g, comps, meta)。"""
    P = np.asarray(P, float)
    L = P.shape[0]
    assert P.ndim == 2 and P.shape[1] == 2, f"P must be (L,2), got {P.shape}"
    active_mask = np.asarray(active_mask, bool)
    assert active_mask.shape == (L,), f"active_mask shape must be (L,), got {active_mask.shape}"

    # 场景副本（仅添加/覆盖 _active_ids 与 _ext_dir）
    sc = dict(scene or {})
    ids = _as_active_ids(active_mask)
    sc["_active_ids"] = ids
    sc["_active_ids_solver"] = ids

    E_total = 0.0
    g = np.zeros_like(P)      # ∇E
    comps: Dict[str, Array2] = {}
    sources_merged: Dict[str, Any] = {}
    wmap = _get_weights(cfg)

    # —— 阶段 1：pre_anchor —— #
    Fsum_ext = np.zeros_like(P)
    pre_terms = list(_eng_enabled(cfg, phase="pre_anchor"))
    for name in pre_terms:
        # 在 compute/forces 优先；否则回退引擎 term
        fn = _CMP_REG.get(name) or _ENG_REG.get(name)
        if fn is None:
            continue
        E_i, F_i, src = fn(sc, P, cfg, phase="pre_anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        if F_i.shape != P.shape:
            raise ValueError(f"[TERM SHAPE MISMATCH] term={name} F={F_i.shape} P={P.shape}")
        F_i = _clip_if_needed(F_i, cfg)

        w = _weight_for(name, wmap)
        E_total += w * float(E_i)
        Fi_w = w * F_i
        comps[name] = Fi_w
        Fsum_ext += Fi_w
        if src:
            merge_sources(sources_merged, src)

    # —— 阶段 2：anchor —— #
    sc["_ext_dir"] = Fsum_ext.copy()
    anchor_terms = list(_eng_enabled(cfg, phase="anchor"))
    for name in anchor_terms:
        fn = _CMP_REG.get(name) or _ENG_REG.get(name)
        if fn is None:
            continue
        E_i, F_i, src = fn(sc, P, cfg, phase="anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        if F_i.shape != P.shape:
            raise ValueError(f"[TERM SHAPE MISMATCH] term={name} F={F_i.shape} P={P.shape}")
        F_i = _clip_if_needed(F_i, cfg)

        w = _weight_for(name, wmap)
        E_total += w * float(E_i)
        comps[name] = w * F_i
        if src:
            merge_sources(sources_merged, src)

    # ∇E = - sum F
    for F_i in comps.values():
        g -= F_i

    # 非活跃行为 0（双保险）
    g[~active_mask] = 0.0
    for k in list(comps.keys()):
        Fi = comps[k]
        Fi[~active_mask] = 0.0
        comps[k] = Fi

    meta = {"sources": sources_merged}
    return float(E_total), g, comps, meta


def energy_and_grad_full(P: Array2, scene, active_mask: np.ndarray, cfg: dict):
    """Compute energy, gradient, and per-term forces via compute aggregator."""
    return _energy_and_grad_full_compute(P, scene, active_mask, cfg)
