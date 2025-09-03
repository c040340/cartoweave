import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from cartoweave.viz.panels import draw_field_panel


def test_draw_field_panel_3d_respects_frame_ratio():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    field = np.arange(4, dtype=float).reshape(2, 2)
    draw_field_panel(ax, field, width=192, height=108, kind="3d")
    aspect = ax.get_box_aspect()
    assert aspect[0] / aspect[1] == pytest.approx(192 / 108)


def test_draw_field_panel_3d_caps_z_aspect():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    field = np.array([[0.0, 0.0], [0.0, 1e6]])
    draw_field_panel(ax, field, width=192, height=108, kind="3d")
    aspect = ax.get_box_aspect()
    assert aspect[2] == pytest.approx(aspect[0])
