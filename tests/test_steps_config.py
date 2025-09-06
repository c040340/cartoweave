import numpy as np

from cartoweave.data.api import build_solvepack_direct
from cartoweave.data.pack_utils import steps_to_stages


def _stage_increments(stages, n):
    prev = np.zeros(n, dtype=bool)
    out = []
    for st in stages:
        mask = st.get("mask_override", np.ones(n, bool))
        out.append(int((mask & ~prev).sum()))
        prev = mask
    return out


def test_steps_sequential_equals_labels():
    n = 5
    stages = steps_to_stages({"kind": "sequential", "steps": 5}, n)
    assert len(stages) == 5  # noqa: S101
    assert _stage_increments(stages, n) == [1, 1, 1, 1, 1]  # noqa: S101


def test_steps_sequential_less_than_labels():
    n = 5
    stages = steps_to_stages({"kind": "sequential", "steps": 3}, n)
    assert len(stages) == 3  # noqa: S101
    assert _stage_increments(stages, n) == [1, 1, 3]  # noqa: S101


def test_steps_group_sizes():
    n = 5
    stages = steps_to_stages({"kind": "grouped", "group_sizes": [2, 1, 2]}, n)
    assert len(stages) == 3  # noqa: S101
    assert _stage_increments(stages, n) == [2, 1, 2]  # noqa: S101


def test_anchors_modes_passthrough():
    sp = build_solvepack_direct(
        frame_size=(100, 100),
        n_labels=4,
        n_points=0,
        n_lines=1,
        n_areas=1,
        anchors_policy={"line": 1.0, "area": 1.0},
        anchor_modes={"line": "midpoint", "area": "centroid"},
        seed=1,
    )
    labels = sp.scene["labels"]
    assert any(lbl["anchor_kind"] == "line" for lbl in labels)  # noqa: S101,E741
    assert any(lbl["anchor_kind"] == "area" for lbl in labels)  # noqa: S101,E741
    for lab in labels:
        if lab["anchor_kind"] == "line":
            assert lab.get("mode") == "midpoint"  # noqa: S101
        if lab["anchor_kind"] == "area":
            assert lab.get("mode") == "centroid"  # noqa: S101

