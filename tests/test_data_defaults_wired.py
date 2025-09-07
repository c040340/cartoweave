import numpy as np

from cartoweave.config.loader import load_data_defaults
from cartoweave.data.generate import generate_scene


def test_load_data_defaults_ok():
    cfg = load_data_defaults("configs/data.yaml")
    assert cfg.source == "generate"
    assert cfg.generate is not None
    assert cfg.generate.counts.points >= 0
    assert cfg.generate.route_gen.segment_len_scale > 0


def test_generate_scene_shapes_and_ranges():
    cfg = load_data_defaults("configs/data.yaml")
    assert cfg.generate is not None
    rng = np.random.default_rng(1)
    P0, labels0, active0, scene0 = generate_scene(cfg.generate, rng)
    assert P0.shape[1] == 2
    total = (
        cfg.generate.counts.points
        + cfg.generate.counts.lines
        + cfg.generate.counts.areas
    )
    assert len(labels0) == total
    for lbl in labels0:
        if lbl.kind == "line" and lbl.polyline is not None:
            assert len(lbl.polyline) >= cfg.generate.shapes.line_min_vertices
