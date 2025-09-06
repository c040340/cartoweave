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

from .forces import REGISTRY as _CMP_REG, enabled_terms as _cmp_enabled

Array2 = np.ndarray


def _as_active_ids(mask: np.ndarray) -> List[int]:
    mask = np.asarray(mask, bool)
    return np.where(mask)[0].tolist()
def energy_and_grad_full(
    P: Array2,
    labels: Any,
    scene: Dict[str, Any],
    active_mask: np.ndarray,
    cfg: dict,
) -> Tuple[float, Array2, Dict[str, Array2]]:
    """Aggregate energy and gradient across all enabled force terms"""
    P = np.asarray(P, float)
    active_mask = np.asarray(active_mask, bool)

    sc = dict(scene or {})
    if labels is not None:
        sc["labels"] = labels
    else:
        sc.setdefault("labels", [])
    ids = _as_active_ids(active_mask)
    sc["_active_ids"] = ids
    sc["_active_ids_solver"] = ids

    E_total = 0.0
    g = np.zeros_like(P)
    comps: Dict[str, Array2] = {}

    # Phase 1: pre_anchor
    Fsum_ext = np.zeros_like(P)
    for name in _cmp_enabled(cfg, phase="pre_anchor"):
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        E_i, F_i, _ = fn(sc, P, cfg, phase="pre_anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        E_total += float(E_i)
        comps[name] = F_i
        Fsum_ext += F_i

    # Phase 2: anchor
    sc["_ext_dir"] = Fsum_ext.copy()
    for name in _cmp_enabled(cfg, phase="anchor"):
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        E_i, F_i, _ = fn(sc, P, cfg, phase="anchor")
        if F_i is None:
            F_i = np.zeros_like(P)
        F_i = np.asarray(F_i, float)
        E_total += float(E_i)
        comps[name] = F_i

    # ∇E = -ΣF
    for F_i in comps.values():
        g -= F_i

    # mask inactive rows
    g[~active_mask] = 0.0
    for k, Fi in comps.items():
        Fi = Fi.copy()
        Fi[~active_mask] = 0.0
        comps[k] = Fi

    return float(E_total), g, comps
