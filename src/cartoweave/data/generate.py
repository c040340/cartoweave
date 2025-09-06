"""Programmatic generators for synthetic scenes and behavior timelines."""

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np

from cartoweave.contracts.solvepack import (
    Scene,
    LabelState,
    AnchorSpec,
    Behavior,
    BehaviorOp,
)

__all__ = [
    "generate_scene",
    "generate_labels",
    "assign_anchor",
    "generate_behaviors",
]


# ---------------------------------------------------------------------------
#  Scene and label generation
# ---------------------------------------------------------------------------


def generate_scene(
    num_points: int,
    num_lines: int,
    num_areas: int,
    frame_size: Tuple[float, float] = (1920.0, 1080.0),
    seed: int = 0,
) -> Scene:
    """Generate a simple synthetic scene.

    The repository ships with sophisticated random scene builders under
    ``data/build_random``.  For the purposes of this refactor a lightweight
    fallback based on uniform sampling suffices; should those utilities be
    present they can easily be swapped in.
    """

    rng = np.random.default_rng(seed)
    W, H = float(frame_size[0]), float(frame_size[1])

    points = rng.uniform([0, 0], [W, H], size=(int(num_points), 2))

    lines: List[np.ndarray] = []
    for _ in range(int(num_lines)):
        start = rng.uniform([0, 0], [W, H], size=2)
        end = rng.uniform([0, 0], [W, H], size=2)
        lines.append(np.vstack([start, end]))

    areas: List[Dict[str, np.ndarray]] = []
    for _ in range(int(num_areas)):
        cx, cy = rng.uniform([0, 0], [W, H])
        w, h = rng.uniform(0.05, 0.2, size=2) * np.array([W, H])
        x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x1, y1 = min(W, cx + w / 2), min(H, cy + h / 2)
        exterior = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        areas.append({"exterior": exterior, "holes": []})

    return Scene(points=points, lines=lines, areas=areas, frame_size=(W, H))


def generate_labels(
    N: int,
    scene: Scene,
    behavior_cfg: Dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[LabelState]]:
    """Initial label state: no active labels and zero sizes/positions."""

    P0 = np.zeros((N, 2), dtype=float)
    active0 = np.zeros(N, dtype=bool)
    labels0 = [
        LabelState(kind="none", WH=np.zeros(2, float), anchor=None, meta={})
        for _ in range(N)
    ]
    return P0, active0, labels0


# ---------------------------------------------------------------------------
#  Behavior helpers
# ---------------------------------------------------------------------------


def assign_anchor(
    i: int, scene: Scene, policy: str = "round_robin", offset: int = 0
) -> AnchorSpec | None:
    """Return an anchor in round-robin fashion across points/lines/areas."""

    if policy != "round_robin":  # pragma: no cover - only policy implemented
        return None

    pts = int(len(scene.points))
    lines = int(len(scene.lines))
    areas = int(len(scene.areas))
    total = pts + lines + areas
    if total == 0:
        return None
    j = (i + offset) % total
    if j < pts:
        return AnchorSpec("point", j)
    j -= pts
    if j < lines:
        return AnchorSpec("line", j)
    j -= lines
    if j < areas:
        return AnchorSpec("area", j)
    return None


def generate_behaviors(
    N: int,
    S: int,
    scene: Scene,
    behavior_cfg: Dict | None = None,
    policy: str = "round_robin",
    seed: int = 0,
) -> List[Behavior]:
    """Create a deterministic behavior timeline.

    Activation pattern: for each step ``k`` the label ``i = k % N`` is targeted.
    If inactive, it is activated and assigned an anchor; otherwise resize and
    deactivate alternatingly.
    """

    rng = np.random.default_rng(seed)
    default_WH = (behavior_cfg or {}).get(
        "default_WH",
        {"point": [8.0, 8.0], "line": [12.0, 6.0], "area": [40.0, 30.0], "none": [0.0, 0.0]},
    )

    active = np.zeros(N, dtype=bool)
    states = [LabelState(kind="none", WH=np.zeros(2), anchor=None, meta={}) for _ in range(N)]
    resized = [False] * N
    behaviors: List[Behavior] = []
    anchor_cursor = 0

    for k in range(int(S)):
        i = k % N
        ops: Dict[str, List] = {"activate": [], "deactivate": [], "mutate": []}
        if not active[i]:
            anc = assign_anchor(anchor_cursor, scene, policy)
            anchor_cursor += 1
            kind = anc.kind if anc else "none"
            WH = default_WH.get(kind, default_WH["point"])
            ops["activate"].append(i)
            mut = {"id": i, "set": {"kind": kind, "WH": WH}}
            if anc is not None:
                mut["set"]["anchor"] = {"kind": anc.kind, "index": anc.index}
            ops["mutate"].append(mut)
            active[i] = True
            resized[i] = False
            states[i].kind = kind
            states[i].WH = np.asarray(WH, float)
            states[i].anchor = anc
        else:
            if not resized[i]:
                new_WH = (states[i].WH * 0.85).tolist()
                ops["mutate"].append({"id": i, "set": {"WH": new_WH}})
                states[i].WH = np.asarray(new_WH, float)
                resized[i] = True
            else:
                ops["deactivate"].append(i)
                active[i] = False
                resized[i] = False
        behaviors.append(Behavior(iters=5, ops=ops, solver="lbfgs", params={}))

    return behaviors

