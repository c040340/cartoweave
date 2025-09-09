import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from cartoweave.viz.layout_style import LayoutStyle
from cartoweave.viz.panels import draw_layout


def _point_count(ax):
    return len(ax.collections)


def _line_count(ax):
    return len(ax.lines)


def _area_count(ax):
    return sum(isinstance(p, PathPatch) for p in ax.patches)


def test_geometry_visibility_controls():
    style = LayoutStyle(show_grid=False)
    pos = np.array([[0.0, 0.0]])
    wh = np.array([[1.0, 1.0]])
    fig, ax = plt.subplots()

    # Points
    points = np.array([[1.0, 1.0]])
    labels = [{"anchor": {"target": "point", "index": 0}}]
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        points=points,
        active_mask=[True],
        style=style,
    )
    assert _point_count(ax) == 1
    ax.cla()
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        points=points,
        active_mask=[False],
        style=style,
    )
    assert _point_count(ax) == 0
    ax.cla()
    draw_layout(
        ax,
        pos[:0],
        [],
        wh[:0],
        frame_w=10,
        frame_h=10,
        points=points,
        active_mask=[],
        style=style,
    )
    assert _point_count(ax) == 0

    # Lines
    ax.cla()
    lines = [np.array([[0, 0], [1, 1]])]
    labels = [{"anchor": {"target": "line", "index": 0}}]
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        lines=lines,
        active_mask=[True],
        style=style,
    )
    assert _line_count(ax) == 1
    ax.cla()
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        lines=lines,
        active_mask=[False],
        style=style,
    )
    assert _line_count(ax) == 0
    ax.cla()
    draw_layout(
        ax,
        pos[:0],
        [],
        wh[:0],
        frame_w=10,
        frame_h=10,
        lines=lines,
        active_mask=[],
        style=style,
    )
    assert _line_count(ax) == 0

    # Areas
    ax.cla()
    areas = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]
    labels = [{"anchor": {"target": "area", "index": 0}}]
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        areas=areas,
        active_mask=[True],
        style=style,
    )
    assert _area_count(ax) == 1
    ax.cla()
    draw_layout(
        ax,
        pos,
        labels,
        wh,
        frame_w=10,
        frame_h=10,
        areas=areas,
        active_mask=[False],
        style=style,
    )
    assert _area_count(ax) == 0
    ax.cla()
    draw_layout(
        ax,
        pos[:0],
        [],
        wh[:0],
        frame_w=10,
        frame_h=10,
        areas=areas,
        active_mask=[],
        style=style,
    )
    assert _area_count(ax) == 0

    plt.close(fig)
