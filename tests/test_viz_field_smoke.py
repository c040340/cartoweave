import os
os.environ["MPLBACKEND"] = "Agg"  # pragma: no cover

import numpy as np
import matplotlib.pyplot as plt

from cartoweave.viz.panels import draw_field_panel


def test_field_panel_smoke():
    fig, ax = plt.subplots()
    width, height = 100.0, 50.0
    field = np.ones((5, 10))
    draw_field_panel(ax, field, width, height, "heatmap", "viridis")
    assert ax.images, "field image should be rendered"
    img = ax.images[0]
    assert tuple(img.get_extent()) == (0.0, width, height, 0.0)
    assert img.get_array().size > 0
    plt.close(fig)
