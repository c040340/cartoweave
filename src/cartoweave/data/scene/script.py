"""Scene script builder producing appear/change/hide events."""
from __future__ import annotations

from typing import List, Dict

import numpy as np

ALLOWED_MODES = ("appear", "change", "hide")


def build_scene_script(
    rng: np.random.Generator,
    n_labels: int,
    n_steps: int,
    ensure_every_label_appears: bool = True,
) -> List[Dict]:
    """Build a simple timeline of label events.

    Each event is a dict with keys ``step`` (int), ``mode`` and ``label``.
    """
    appear_steps = rng.integers(0, max(1, n_steps), size=n_labels)
    if ensure_every_label_appears and n_steps > 0:
        appear_steps = np.clip(appear_steps, 0, n_steps - 1)
    events: List[Dict] = []
    appeared: set[int] = set()
    active: set[int] = set()

    for step in range(n_steps):
        for lid in range(n_labels):
            if lid not in appeared and appear_steps[lid] == step:
                events.append({"step": step, "mode": "appear", "label": lid})
                appeared.add(lid)
                active.add(lid)
        for lid in list(active):
            r = rng.random()
            if r < 0.1:
                events.append({"step": step, "mode": "change", "label": lid})
            elif r < 0.15:
                events.append({"step": step, "mode": "hide", "label": lid})
                active.remove(lid)
    if ensure_every_label_appears:
        for lid in range(n_labels):
            if lid not in appeared:
                events.append({"step": n_steps, "mode": "appear", "label": lid})
    return events

