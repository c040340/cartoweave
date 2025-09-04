import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import asdict
from pathlib import Path
from cartoweave.config_loader import load_configs
from cartoweave.viz.panels import draw_layout
from cartoweave.viz.defaults import merge_defaults


def test_anchor_marker_size_from_yaml(tmp_path):
    orig = Path("configs/viz.yaml").read_text()
    viz_path = tmp_path / "viz.yaml"
    viz_path.write_text(orig.replace("anchor_marker_size: 4.0", "anchor_marker_size: 11.0"))
    bundle = load_configs(viz_path=str(viz_path), run_path=str(tmp_path / "no_run.yaml"))
    viz = merge_defaults(asdict(bundle.viz))
    fig, ax = plt.subplots()
    draw_layout(
        ax,
        pos=np.zeros((1, 2)),
        labels=[{}],
        rect_wh=np.array([[10.0, 10.0]]),
        frame_w=100.0,
        frame_h=100.0,
        anchors=np.array([[5.0, 5.0]]),
        viz_layout=viz["layout"],
    )
    from matplotlib.patches import Circle

    circles = [p for p in ax.patches if isinstance(p, Circle)]
    radius = circles[0].radius
    assert radius == 11.0
