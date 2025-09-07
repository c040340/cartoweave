from __future__ import annotations

from typing import List

import numpy as np

from cartoweave.contracts.solvepack import ActionRecord

__all__ = ["generate_action_sequence_strict"]


def generate_action_sequence_strict(L: int, S: int, rng: np.random.Generator) -> List[ActionRecord]:
    """Generate a strict sequence of ``S`` actions for ``L`` labels.

    Each label receives at least one ``appear`` when ``S >= L`` and additional
    actions are distributed uniformly at random.  Action timestamps are sampled
    in ``(0,1)`` and globally ordered with strict increase.
    """
    if S < 0:
        raise ValueError("S must be >=0")
    if S == 0:
        return []
    if L <= 0:
        raise ValueError("L must be >0 when S>0")

    if S <= L:
        counts = np.zeros(L, dtype=int)
        ids = rng.choice(L, size=S, replace=False)
        counts[ids] = 1
    else:
        counts = np.ones(L, dtype=int)
        for _ in range(S - L):
            j = int(rng.integers(0, L))
            counts[j] += 1

    raw_actions: list[dict] = []
    eps = 1e-9
    for lbl_id, a in enumerate(counts):
        if a == 0:
            continue
        t1 = float(rng.uniform(0.0, 1.0 - eps))
        times = [t1]
        if a > 1:
            rest = rng.uniform(t1, 1.0, size=a - 1)
            rest.sort()
            times.extend(float(x) for x in rest)
        types = ["appear"]
        if a == 2:
            types.append("mutate" if rng.random() < 0.5 else "disappear")
        elif a >= 3:
            types.extend(["mutate"] * (a - 2))
            types.append("mutate" if rng.random() < 0.5 else "disappear")
        for t, tp in zip(times, types):
            raw_actions.append({"t": float(t), "step": None, "id": lbl_id, "type": tp})

    raw_actions.sort(key=lambda a: a["t"])
    last = -1.0
    for a in raw_actions:
        if a["t"] <= last:
            a["t"] = last + eps
        last = a["t"]
    actions = [ActionRecord.model_validate(a) for a in raw_actions]
    assert len(actions) == S, f"expected {S} actions, got {len(actions)}"
    return actions
