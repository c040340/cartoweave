from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["steps_to_stages", "active_mask0_from_stages"]


def steps_to_stages(steps_cfg: dict[str, Any], n_labels: int, per_step_iters: int = 6) -> list[dict[str, Any]]:
    """Convert *steps_cfg* into a list of solver ``stages``.

    Parameters
    ----------
    steps_cfg:
        Dict with keys ``kind`` and ``steps`` and optionally ``group_sizes`` or
        ``groups`` as described in the canonical schema.
    n_labels:
        Number of labels.
    per_step_iters:
        Iterations per stage (default ``6``).
    """
    kind = steps_cfg.get("kind", "none")
    stages: list[dict[str, Any]] = []

    if kind == "none":
        stages.append({"iters": per_step_iters})
        return stages

    if kind == "sequential":
        steps = steps_cfg.get("steps")
        if steps is None:
            steps = n_labels
        steps = max(1, int(steps))
        active = np.zeros(n_labels, dtype=bool)
        for s in range(steps):
            if s < steps - 1:
                if s < n_labels:
                    active[s] = True
            else:
                active[:] = True
            stages.append({"iters": per_step_iters, "mask_override": active.copy()})
        return stages

    if kind == "grouped":
        groups = steps_cfg.get("groups")
        group_sizes = steps_cfg.get("group_sizes")
        groups_list: list[list[int]] = []
        if groups:
            groups_list = [list(g) for g in groups]
        elif group_sizes:
            idx = 0
            for sz in group_sizes:
                next_idx = min(n_labels, idx + int(sz))
                groups_list.append(list(range(idx, next_idx)))
                idx = next_idx
            if groups_list and idx < n_labels:
                groups_list[-1].extend(range(idx, n_labels))
        else:
            base = n_labels // 3
            rem = n_labels % 3
            sizes = [base + (1 if i < rem else 0) for i in range(3)]
            idx = 0
            for sz in sizes:
                groups_list.append(list(range(idx, idx + sz)))
                idx += sz
        active = np.zeros(n_labels, dtype=bool)
        for g in groups_list:
            active[g] = True
            stages.append({"iters": per_step_iters, "mask_override": active.copy()})
        return stages

    raise ValueError(f"unknown steps.kind: {kind}")


def active_mask0_from_stages(stages: list[dict[str, Any]], n_labels: int) -> np.ndarray:
    if stages and "mask_override" in stages[0]:
        return np.array(stages[0]["mask_override"], dtype=bool)
    return np.ones(n_labels, dtype=bool)
