import numpy as np
import pytest

from cartoweave.compute.sources import make_sources_from_scene
from cartoweave.compute.types import VPSources


def test_make_sources_from_scene_success():
    scene = {
        "frame_size": (100, 80),
        "points": [(0.0, 0.0), (1.0, 2.0)],
        "lines": [[(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)]],
        "areas": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],
        "circles": [{"xy": (5.0, 5.0), "r": 2.0}],
    }
    src = make_sources_from_scene(scene)
    assert isinstance(src, VPSources)
    assert src.points.shape == (2, 2)
    assert len(src.lines) == 1 and src.lines[0].shape == (3, 2)
    assert len(src.areas) == 2
    assert {a["kind"] for a in src.areas} == {"poly", "circle"}
    assert src.frame_size == (100, 80)


def test_make_sources_from_scene_nonfinite():
    scene = {"frame_size": (10, 10), "points": [(0.0, 0.0), (np.nan, 1.0)]}
    with pytest.raises(ValueError):
        make_sources_from_scene(scene)


def test_make_sources_from_scene_flattened_line():
    scene = {"frame_size": (10, 10), "points": [], "lines": [[0.0, 0.0, 1.0, 1.0]]}
    with pytest.raises(ValueError):
        make_sources_from_scene(scene)
