import numpy as np
from cartoweave.data.scene.make import get_scene


def test_get_scene_deterministic():
    s1 = get_scene((200, 100), seed=123)
    s2 = get_scene((200, 100), seed=123)
    assert np.allclose(s1["points"], s2["points"])
    assert len(s1["lines"]) == len(s2["lines"])
    assert len(s1["areas"]) == len(s2["areas"])
    assert np.allclose(s1["labels_init"], s2["labels_init"])
    assert np.allclose(s1["WH"], s2["WH"])
    for key in ["points", "lines", "areas", "labels_init", "WH", "script"]:
        assert key in s1
