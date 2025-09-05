import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from cartoweave.data.random import generate_scene
from cartoweave.viz import view


def _capture(scene, monkeypatch):
    captured = {}

    def spy(ax, pos, labels, rect_wh, **kwargs):
        captured["points"] = kwargs.get("points")
        captured["lines"] = kwargs.get("lines")
        captured["areas"] = kwargs.get("areas")
        return []

    monkeypatch.setattr(view, "draw_layout", spy)
    monkeypatch.setattr(view, "draw_force_panel", lambda *a, **k: np.zeros(2))
    monkeypatch.setattr(view, "draw_field_panel", lambda *a, **k: None)
    monkeypatch.setattr(view, "select_terms_for_arrows", lambda *a, **k: [])
    monkeypatch.setattr(view, "normalize_comps_for_info", lambda *a, **k: {})
    monkeypatch.setattr(view.plt, "show", lambda *a, **k: None)

    traj = scene["labels_init"][None, :, :]
    rect_wh = scene["WH"]
    labels = scene["labels"]
    W, H = scene["frame_size"]

    view.interactive_view(
        traj,
        labels,
        rect_wh,
        scene.get("points"),
        scene.get("lines"),
        scene.get("areas"),
        W,
        H,
        force_getter=lambda step: {},
        source_getter=lambda step: {},
    )
    return captured


def test_lines_are_vec2(monkeypatch):
    scene = generate_scene(canvas_size=(200, 100), n_points=0, n_lines=1, n_areas=0, seed=1)
    cap = _capture(scene, monkeypatch)
    lines = cap.get("lines")
    assert isinstance(lines, list)
    assert len(lines) == 1
    for pl in lines:
        arr = np.asarray(pl, float)
        assert arr.ndim == 2 and arr.shape[1] == 2


def test_areas_are_vec2(monkeypatch):
    scene = generate_scene(canvas_size=(200, 100), n_points=0, n_lines=0, n_areas=1, seed=2)
    cap = _capture(scene, monkeypatch)
    areas = cap.get("areas")
    assert isinstance(areas, list)
    assert len(areas) == 1
    for poly in areas:
        arr = np.asarray(poly, float)
        assert arr.ndim == 2 and arr.shape[1] == 2


def test_points_batching(monkeypatch):
    scene = generate_scene(canvas_size=(200, 100), n_points=3, n_lines=0, n_areas=0, seed=3)
    cap = _capture(scene, monkeypatch)
    pts = np.asarray(cap.get("points"), float)
    assert pts.ndim == 2 and pts.shape == (3, 2)


def test_no_flatten_patterns():
    import re
    from pathlib import Path

    files = [
        Path("src/cartoweave/data/random.py"),
        Path("src/cartoweave/compute/forces/line_label.py"),
        Path("src/cartoweave/labels.py"),
    ]
    pat = re.compile(r"\.flatten\(|\.ravel\(|reshape\(-1")
    for f in files:
        txt = f.read_text()
        assert not pat.search(txt), f"flatten pattern found in {f}"


def test_as_vec2_strict():
    from cartoweave.viz.view import _as_vec2

    arr = np.array([1.0, 2.0, 3.0, 4.0], float)
    assert _as_vec2(arr) is None
