# ruff: noqa: S101, N806
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
    P0, labels0, scene0 = generate_scene(cfg.generate, rng)
    assert P0.shape[1] == 2
    counts = cfg.generate.counts
    assert len(scene0.points) == counts.points
    assert len(scene0.lines) == counts.lines
    assert len(scene0.areas) == counts.areas
    expect_labels = cfg.generate.labels or (counts.points + counts.lines + counts.areas)
    assert len(labels0) == expect_labels
    for lbl in labels0:
        assert not hasattr(lbl, "polyline")
        assert not hasattr(lbl, "polygon")
