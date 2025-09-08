import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
import yaml
from cartoweave.viz.panels import draw_layout
from cartoweave.viz.defaults import merge_defaults
from cartoweave.viz.layout_style import get_layout_style_from_cfg


def test_anchor_marker_size_from_yaml(tmp_path):
    orig = Path("configs/viz.yaml").read_text()
    viz_path = tmp_path / "viz.yaml"
    viz_path.write_text(orig.replace("anchor_marker_size: 4.0", "anchor_marker_size: 11.0"))
    viz_cfg = yaml.safe_load(viz_path.read_text())
    viz = merge_defaults(viz_cfg)
    assert viz["layout"]["colors"]["points"] == "#2E86DE"
    style = get_layout_style_from_cfg(viz)
    fig, ax = plt.subplots()
    draw_layout(
        ax,
        pos=np.zeros((1, 2)),
        labels=[{}],
        rect_wh=np.array([[10.0, 10.0]]),
        frame_w=100.0,
        frame_h=100.0,
        anchors=np.array([[5.0, 5.0]]),
        style=style,
    )
    from matplotlib.collections import PathCollection

    scat = [c for c in ax.collections if isinstance(c, PathCollection)]
    sizes = scat[0].get_sizes()
    assert sizes[0] == (11.0 * 2) ** 2
