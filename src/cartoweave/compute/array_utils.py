"""Utility helpers for expanding active-subset results to full arrays."""

from __future__ import annotations

from typing import Dict

import numpy as np


def expand_subset(prev_full: np.ndarray, active_idx: np.ndarray, sub_array: np.ndarray) -> np.ndarray:
    """Expand a subset array back to full shape.

    Parameters
    ----------
    prev_full:
        Previous full array of shape (N, 2).
    active_idx:
        Indices of the active rows within the full array.
    sub_array:
        Array containing only the active rows, shape (n_active, 2).

    Returns
    -------
    np.ndarray
        Full array where active rows are replaced by ``sub_array`` and
        inactive rows retain ``prev_full`` values.
    """
    out = np.array(prev_full, dtype=float, copy=True)
    out[np.asarray(active_idx, dtype=int)] = np.asarray(sub_array, dtype=float)
    return out


def expand_comps_subset(
    prev_full: Dict[str, np.ndarray], active_idx: np.ndarray, sub_comps: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Expand a dict of component forces defined on active subset to full arrays.

    Every term in the union of ``prev_full`` and ``sub_comps`` gets a full
    ``(N,2)`` array. Terms missing in ``sub_comps`` are carried over from
    ``prev_full`` (or zeros when unseen).
    """
    out: Dict[str, np.ndarray] = {}
    active_idx = np.asarray(active_idx, dtype=int)

    if prev_full:
        ref = next(iter(prev_full.values()))
        zero_ref = np.zeros_like(ref)
    else:
        # infer size from subset indices; falls back to 0 if empty
        n = int(active_idx.max() + 1) if active_idx.size else 0
        zero_ref = np.zeros((n, 2), float)

    terms = set(prev_full.keys()) | set(sub_comps.keys())
    for k in terms:
        base = prev_full.get(k, zero_ref)
        arr = np.array(base, dtype=float, copy=True)
        if k in sub_comps:
            sub_arr = np.asarray(sub_comps[k], dtype=float)
            if sub_arr.shape == arr.shape:
                arr = sub_arr
            else:
                if sub_arr.shape[0] != active_idx.size:
                    raise IndexError("subset length mismatch for component '%s'" % k)
                arr[active_idx] = sub_arr
        out[k] = arr
    return out


__all__ = ["expand_subset", "expand_comps_subset"]

