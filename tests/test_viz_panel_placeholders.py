import matplotlib
matplotlib.use('Agg')

from cartoweave.viz.vp_view import build_view_axes


def test_build_view_axes_field_placeholder():
    axes = build_view_axes(field_kind="none")
    assert axes.ax_field is not None
