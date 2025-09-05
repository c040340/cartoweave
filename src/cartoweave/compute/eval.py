# -*- coding: utf-8 -*-
"""Compute-only energy and gradient aggregator.

All force terms are evaluated through :mod:`cartoweave.compute.forces` and
assembled into a single energy and gradient. The aggregator runs in two
phases:

``pre_anchor → (inject _ext_dir) → anchor``

For each enabled term a force field ``F_i`` is produced. We ensure shapes are
``(L, 2)``, inactive rows are zeroed, and the gradient satisfies
``G ≈ -Σcomps``.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, List
import numpy as np

from cartoweave.utils.checks import merge_sources
from .forces import REGISTRY as _CMP_REG, enabled_terms as _cmp_enabled

Array2 = np.ndarray


def _as_active_ids(mask: np.ndarray) -> List[int]:
    mask = np.asarray(mask, bool)
    return np.where(mask)[0].tolist()


def _clip_if_needed(F: Array2, cfg: dict) -> Array2:
    clamp_max = (
        cfg.get("compute", {}).get("clamp", {}).get("optimize_force_max")
        or cfg.get("solver", {})
        .get("tuning", {})
        .get("clamp", {})
        .get("optimize_force_max")
    )
    if clamp_max is None:
        return F
    cm = float(clamp_max)
    np.clip(F, -cm, cm, out=F)
    return F


def _energy_and_grad_full_compute(P: Array2, scene: Dict[str, Any], active_mask: np.ndarray, cfg: dict):
    """Evaluate all enabled terms via the compute aggregator.

    Parameters
    ----------
    P, active_mask
        Current positions ``(L,2)`` and boolean mask. Inactive rows are zeroed
        in the output.
    scene, cfg
        Immutable scene data and configuration.

    Returns
    -------
    E_total : float
        Total energy after weighting.
    g : Array2
        Gradient satisfying ``g = -Σ comps``.
    comps : Dict[str, Array2]
        Per-term force fields after weighting. Weights are resolved by exact
        key match, ``prefix.*`` or bare ``prefix``.
    meta : dict
        Aggregated source metadata.
    """
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
    # —— 阶段 1：pre_anchor —— #
    Fsum_ext = np.zeros_like(P)
    pre_terms = list(_cmp_enabled(cfg, phase="pre_anchor"))
    for name in pre_terms:
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        E_i, F_i, src = fn(sc, P, cfg, phase="pre_anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        if F_i.shape != P.shape:
            raise ValueError(f"[TERM SHAPE MISMATCH] term={name} F={F_i.shape} P={P.shape}")
        F_i = _clip_if_needed(F_i, cfg)

        E_total += float(E_i)
        comps[name] = F_i
        Fsum_ext += F_i
        if src:
            merge_sources(sources_merged, src)

    # —— 阶段 2：anchor —— #
    sc["_ext_dir"] = Fsum_ext.copy()
    anchor_terms = list(_cmp_enabled(cfg, phase="anchor"))
    for name in anchor_terms:
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        E_i, F_i, src = fn(sc, P, cfg, phase="anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        if F_i.shape != P.shape:
            raise ValueError(f"[TERM SHAPE MISMATCH] term={name} F={F_i.shape} P={P.shape}")
        F_i = _clip_if_needed(F_i, cfg)

        E_total += float(E_i)
        comps[name] = F_i
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
    """Public entry point for compute-side energy evaluation."""

    return _energy_and_grad_full_compute(P, scene, active_mask, cfg)
