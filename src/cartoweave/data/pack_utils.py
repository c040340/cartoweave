from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

__all__ = ["steps_to_behaviors"]


def _ensure_idx(idx: int, N: int) -> int:
    i = int(idx)
    if not (0 <= i < N):
        raise IndexError(f"index {i} out of range for N={N}")
    return i


def _normalize_mutate(item: Dict[str, Any], N: int) -> Dict[str, Any]:
    idx = _ensure_idx(item.get("id"), N)
    set_dict = item.get("set", {}) or {}
    allowed = {"kind", "WH", "anchor", "meta"}
    out = {k: v for k, v in set_dict.items() if k in allowed}
    return {"id": idx, "set": out}


def steps_to_behaviors(steps_cfg: dict, N: int) -> List[dict]:
    """Translate high-level ``steps_cfg`` into behavior dicts.

    Supports keys like ``appear``, ``disappear``, ``resize``, ``retag`` and
    ``retarget``. Returns a list where each element has the shape::

        {"iters": int,
         "ops": {"activate": [...], "deactivate": [...], "mutate": [...]},
         "solver": str, "params": {}}

    Legacy keys such as ``mask_override`` or ``stages`` raise ``ValueError``.
    Indices are validated to lie in ``[0, N)``. Missing fields default to empty
    ops and solver ``"lbfgs"``.
    """

    legacy = {"mask_override", "stages"}
    if any(k in steps_cfg for k in legacy):
        raise ValueError("Legacy steps detected: migrate to behaviors.")

    kind = steps_cfg.get("kind")
    iters_default = int(steps_cfg.get("iters", 6))
    behaviors: List[dict] = []

    if kind == "none" or (kind is None and not steps_cfg.get("script")):
        behaviors.append({"iters": iters_default, "ops": {"activate": [], "deactivate": [], "mutate": []}, "solver": "lbfgs", "params": {}})
        return behaviors

    if kind == "sequential":
        steps = steps_cfg.get("steps")
        if steps is None:
            steps = N
        steps = max(1, int(steps))
        active = np.zeros(N, dtype=bool)
        for s in range(steps):
            if s < steps - 1:
                ids = [s] if s < N else []
            else:
                ids = [i for i in range(N) if not active[i]]
            behaviors.append({"iters": iters_default, "ops": {"activate": ids, "deactivate": [], "mutate": []}, "solver": "lbfgs", "params": {}})
            for i in ids:
                if 0 <= i < N:
                    active[i] = True
        return behaviors

    if kind == "grouped":
        groups = steps_cfg.get("groups")
        group_sizes = steps_cfg.get("group_sizes")
        groups_list: List[List[int]] = []
        if groups:
            groups_list = [[_ensure_idx(i, N) for i in g] for g in groups]
        elif group_sizes:
            idx = 0
            for sz in group_sizes:
                next_idx = min(N, idx + int(sz))
                groups_list.append(list(range(idx, next_idx)))
                idx = next_idx
            if groups_list and idx < N:
                groups_list[-1].extend(range(idx, N))
        else:
            base = N // 3
            rem = N % 3
            sizes = [base + (1 if i < rem else 0) for i in range(3)]
            idx = 0
            for sz in sizes:
                groups_list.append(list(range(idx, min(N, idx + sz))))
                idx += sz
        active = np.zeros(N, dtype=bool)
        for g in groups_list:
            ids = [i for i in g if not active[i]]
            behaviors.append({"iters": iters_default, "ops": {"activate": ids, "deactivate": [], "mutate": []}, "solver": "lbfgs", "params": {}})
            active[g] = True
        return behaviors

    script = steps_cfg.get("script") or steps_cfg.get("timeline") or []
    for step in script:
        if any(k in step for k in legacy):
            raise ValueError("Legacy steps detected: migrate to behaviors.")
        ops = step.get("ops") or {}
        act: List[int] = []
        deact: List[int] = []
        mut: List[Dict[str, Any]] = []

        if "appear" in step:
            act = [_ensure_idx(i, N) for i in step["appear"]]
        if "disappear" in step:
            deact = [_ensure_idx(i, N) for i in step["disappear"]]
        if "resize" in step:
            mut.extend(_normalize_mutate({"id": m.get("id"), "set": {"WH": m.get("WH")}}, N) for m in step["resize"])
        if "retag" in step:
            mut.extend(_normalize_mutate({"id": m.get("id"), "set": {"kind": m.get("kind")}}, N) for m in step["retag"])
        if "retarget" in step:
            mut.extend(_normalize_mutate({"id": m.get("id"), "set": {"anchor": m.get("anchor")}}, N) for m in step["retarget"])
        if ops:
            act.extend(_ensure_idx(i, N) for i in ops.get("activate", []))
            deact.extend(_ensure_idx(i, N) for i in ops.get("deactivate", []))
            mut.extend(_normalize_mutate(m, N) for m in ops.get("mutate", []))

        beh = {
            "iters": int(step.get("iters", iters_default)),
            "ops": {"activate": act, "deactivate": deact, "mutate": mut},
            "solver": step.get("solver", "lbfgs"),
            "params": step.get("params", {}),
        }
        behaviors.append(beh)
    return behaviors
