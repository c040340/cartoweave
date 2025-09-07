from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import ImageFont

from cartoweave.contracts.solvepack import ActionRecord, Label
from cartoweave.data.textblock import measure_text_block, random_text_lines

__all__ = ["generate_action_sequence_strict"]


def generate_action_sequence_strict(
    labels0: List[Label],
    S: int,
    rng: np.random.Generator,
    font: ImageFont.FreeTypeFont,
    len_range: Tuple[int, int],
    line_spacing_px: int,
    padding_x: int,
    padding_y: int,
    resample_text_on_size_mutate: bool,
) -> List[ActionRecord]:
    """Generate a strict sequence of ``S`` actions for ``labels0``.

    Each label receives at least one ``appear`` when ``S >= len(labels0)`` and
    additional actions are distributed uniformly at random.  Action timestamps
    are sampled in ``(0,1)`` and globally ordered with strict increase.  ``WH_to``
    is populated for ``appear`` and ``mutate`` actions based on Pillow
    measurement.
    """
    L = len(labels0)
    if S < 0:
        raise ValueError("S must be >=0")
    if S == 0:
        return []
    if L <= 0:
        raise ValueError("labels0 must be non-empty when S>0")

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
            act: dict = {"t": float(t), "step": None, "id": lbl_id, "type": tp}
            if tp == "appear":
                act["WH_to"] = labels0[lbl_id].WH
                act["kind_to"] = "rectangular"
            elif tp == "mutate":
                if resample_text_on_size_mutate:
                    lines = random_text_lines(rng, len_range)
                    W, H = measure_text_block(lines, font, line_spacing_px, padding_x, padding_y)
                    act["WH_to"] = (float(W), float(H))
                else:
                    act["WH_to"] = labels0[lbl_id].WH
            raw_actions.append(act)

    raw_actions.sort(key=lambda a: a["t"])
    last = -1.0
    for a in raw_actions:
        if a["t"] <= last:
            a["t"] = last + eps
        last = a["t"]
    actions = [ActionRecord.model_validate(a) for a in raw_actions]
    assert len(actions) == S, f"expected {S} actions, got {len(actions)}"
    return actions
