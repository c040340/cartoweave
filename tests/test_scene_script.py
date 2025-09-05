from cartoweave.data.scene.script import build_scene_script
import numpy as np


def test_script_invariants():
    rng = np.random.default_rng(0)
    script = build_scene_script(rng, n_labels=5, n_steps=5)
    appeared = set()
    for ev in script:
        lid = ev["label"]
        mode = ev["mode"]
        if mode == "appear":
            assert lid not in appeared
            appeared.add(lid)
        else:
            assert lid in appeared
    appears = [ev["label"] for ev in script if ev["mode"] == "appear"]
    assert len(appears) == len(set(appears))
