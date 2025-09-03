import numpy as np
import matplotlib
import pytest

matplotlib.use('Agg')

from cartoweave.viz.view import interactive_view


@pytest.mark.parametrize(
    "order",
    [
        ("forces", "field", "sources"),
        ("field", "forces", "sources"),
        ("sources", "field", "forces"),
        ("field", "forces"),
    ],
)
def test_force_getter_allows_any_tuple_order(monkeypatch, order):
    # minimal scene with one label and one iteration
    traj = np.zeros((1, 1, 2), dtype=float)
    labels = [{}]
    rect_wh = np.ones((1, 2), dtype=float)
    points = np.zeros((0, 2))
    lines = []
    areas = []

    field = np.arange(4, dtype=float).reshape(2, 2)
    forces = {"t": np.zeros((1, 2), dtype=float)}
    sources = {"dummy": 1}

    captured = {}

    def fake_draw_field(ax, fld, W, H, kind, cmap):
        captured["field"] = fld

    def fake_draw_force(ax, frc, idx, title=None):
        captured["forces"] = frc
        return 0.0, 0.0

    monkeypatch.setattr("cartoweave.viz.view.draw_field_panel", fake_draw_field)
    monkeypatch.setattr("cartoweave.viz.view.draw_force_panel", fake_draw_force)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    items = {"forces": forces, "field": field, "sources": sources}

    def force_getter(step: int):
        return tuple(items[name] for name in order)

    interactive_view(
        traj,
        labels,
        rect_wh,
        points,
        lines,
        areas,
        W=1,
        H=1,
        force_getter=force_getter,
        field_kind="3d",
    )

    assert np.array_equal(captured["field"], field)
    assert captured["forces"] == forces
