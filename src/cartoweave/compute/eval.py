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

from typing import Any

import numpy as np

from .forces import REGISTRY as _CMP_REG
from .forces import enabled_terms as _cmp_enabled, term_params_map
from .geom_anchor_resolver import anchor_position

Array2 = np.ndarray
def energy_and_grad_full(
    P: Array2,  # noqa: N803
    labels: Any,
    scene: dict[str, Any],
    active_mask: np.ndarray,
    cfg: dict,
) -> tuple[float, Array2, dict[str, Array2]]:
    """Aggregate energy and gradient across all enabled force terms"""
    P = np.asarray(P, float)  # noqa: N806
    active_mask = np.asarray(active_mask, bool)

    sc = dict(scene or {})
    if labels is not None:
        sc["labels"] = labels
    else:
        sc.setdefault("labels", [])
    labels_all = sc.get("labels", [])
    sc["anchors"] = np.asarray(
        [anchor_position(labels_all[i], sc, P) for i in range(len(labels_all))], dtype=float
    )

    energy_total = 0.0
    g = np.zeros_like(P)
    comps: dict[str, Array2] = {}

    pmap = term_params_map(cfg)

    # Phase 1: pre_anchor
    fsum_ext = np.zeros_like(P)
    for name in _cmp_enabled(cfg, phase="pre_anchor"):
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        params = pmap.get(name, {})
        k = float(params.get("k", 0.0))
        if k <= 0.0:
            continue
        e_i, f_i, _ = fn(sc, P, params, cfg)
        f_i = np.asarray(f_i if f_i is not None else np.zeros_like(P), float)
        energy_total += k * float(e_i)
        f_scaled = k * f_i
        comps[name] = f_scaled
        fsum_ext += f_scaled

    # Phase 2: anchor
    sc["_ext_dir"] = fsum_ext.copy()
    for name in _cmp_enabled(cfg, phase="anchor"):
        fn = _CMP_REG.get(name)
        if fn is None:
            raise KeyError(f"[TERM MISSING] '{name}' is not registered in compute.forces")
        params = pmap.get(name, {})
        k = float(params.get("k", 0.0))
        if k <= 0.0:
            continue
        e_i, f_i, _ = fn(sc, P, params, cfg)
        f_i = np.asarray(f_i if f_i is not None else np.zeros_like(P), float)
        energy_total += k * float(e_i)
        f_scaled = k * f_i
        comps[name] = f_scaled

    # ∇E = -ΣF
    for f_i in comps.values():
        g -= f_i

    # mask inactive rows
    g[~active_mask] = 0.0
    for k, f_i in comps.items():
        f_i = f_i.copy()
        f_i[~active_mask] = 0.0
        comps[k] = f_i

    return float(energy_total), g, comps
