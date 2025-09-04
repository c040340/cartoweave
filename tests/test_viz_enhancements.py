import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.view import interactive_view
from cartoweave.viz.panels import draw_layout, normalize_comps_for_info, ALL_FORCE_KEYS, select_terms_for_arrows
from cartoweave.config.presets import VIZ_FORCE_CONFIG
from cartoweave.orchestrators.scene_script_runner import run_scene_script
from matplotlib.patches import Circle


def test_step_specific_geometry(monkeypatch):
    points_full = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], float)
    recs = [
        {"P": np.zeros((1, 2)), "comps": {}, "meta": {"sources": {"points": points_full[:2]}}},
        {"P": np.zeros((1, 2)), "comps": {}, "meta": {"sources": {"points": points_full}}},
    ]
    steps = [{"name": "s0", "rec_start": 0, "rec_end": 2, "active_ids_viz": [0]}]
    payload = build_viz_payload({"history": {"records": recs, "scene_steps": steps}})
    frames = payload["frames"]
    traj = np.zeros((2, 1, 2), float)
    labels = [{"anchor_kind": "point", "anchor_index": 0}]
    rect_wh = np.ones((1, 2), float)
    captured = {}

    def fake_draw_layout(ax, pos, labs, wh, frame_w, frame_h, points=None, lines=None, areas=None, anchors=None):
        captured["points"] = points
        return []

    monkeypatch.setattr("cartoweave.viz.view.draw_layout", fake_draw_layout)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    interactive_view(
        traj,
        labels,
        rect_wh,
        points_full,
        [],
        [],
        W=1,
        H=1,
        force_getter=lambda i: {},
        frames=frames,
    )
    assert isinstance(captured["points"], np.ndarray)
    assert captured["points"].shape[0] == 2


def test_geometry_respects_active_ids(monkeypatch):
    points_full = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], float
    )
    recs = [
        {"P": np.zeros((2, 2)), "comps": {}, "meta": {"sources": {"points": points_full}}}
    ]
    steps = [{"name": "s0", "rec_start": 0, "rec_end": 1, "active_ids_viz": [0]}]
    payload = build_viz_payload({"history": {"records": recs, "scene_steps": steps}})
    frames = payload["frames"]
    traj = np.zeros((1, 2, 2), float)
    labels = [
        {"anchor_kind": "point", "anchor_index": 0},
        {"anchor_kind": "point", "anchor_index": 1},
    ]
    rect_wh = np.ones((2, 2), float)
    captured = {}

    def fake_draw_layout(
        ax, pos, labs, wh, frame_w, frame_h, points=None, lines=None, areas=None, anchors=None
    ):
        captured["points"] = points
        return []

    monkeypatch.setattr("cartoweave.viz.view.draw_layout", fake_draw_layout)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    interactive_view(
        traj,
        labels,
        rect_wh,
        points_full,
        [],
        [],
        W=1,
        H=1,
        force_getter=lambda i: {},
        active_getter=lambda i: frames[i].get("active_ids_viz", []),
        frames=frames,
    )
    assert isinstance(captured["points"], np.ndarray)
    assert captured["points"].shape[0] == 1


def test_circle_label_rendering():
    fig, ax = plt.subplots()
    labels = [{"mode": "circle"}]
    pos = np.array([[5.0, 5.0]], float)
    wh = np.array([[4.0, 4.0]], float)
    patches = draw_layout(ax, pos, labels, wh, frame_w=10.0, frame_h=10.0)
    assert isinstance(patches[0][1], Circle)
    assert len(ax.texts) == 0
    plt.close(fig)


def test_normalize_comps_for_info_lists_all():
    N = 2
    comps = {"ll.rect": np.ones((N, 2), float)}
    out = normalize_comps_for_info(comps, N)
    assert set(ALL_FORCE_KEYS).issubset(out.keys())
    for k in ALL_FORCE_KEYS:
        assert out[k].shape == (N, 2)
        if k != "ll.rect":
            assert np.all(out[k] == 0)


def test_arrow_filtering_ratio():
    comps = {
        "a": np.array([[10.0, 0.0]], float),
        "b": np.array([[1.0, 0.0]], float),
        "c": np.array([[0.05, 0.0]], float),
    }
    cfg = dict(VIZ_FORCE_CONFIG)
    cfg["arrows_min_ratio"] = 0.5
    cfg["arrows_min_abs"] = 0.0
    terms = select_terms_for_arrows(comps, cfg)
    assert terms == ["a"]


def test_inactive_force_leak_assert(monkeypatch):
    scene = {
        "labels": [{"id": "a", "visible": True}, {"id": "b", "visible": False}],
        "labels_init": np.zeros((2, 2), float),
        "WH": np.ones((2, 2), float),
    }
    step = {"name": "s0"}

    def fake_run_solve_plan(scene_in, stages, cfg_in):
        record = {"P": np.zeros((1, 2)), "comps": {"fake": np.array([[0.0, 0.0], [1.0, 0.0]])}}
        hist = {"records": [record], "positions": [np.zeros((1, 2))], "energies": [0.0]}
        return {"P_final": np.zeros((1, 2)), "history": hist}

    monkeypatch.setattr(
        "cartoweave.orchestrators.scene_script_runner.run_solve_plan", fake_run_solve_plan
    )
    with pytest.raises(AssertionError):
        run_scene_script(scene, [step], {}, cfg={"DEBUG_VISIBILITY_ASSERTS": True})
