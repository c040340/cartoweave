import os
import numpy as np
from cartoweave.data.random import generate_scene, get_scene, config_hash
from cartoweave import SolvePack, solve

def _tmp_cache(tmp_path, name="scene_cache.npz"):
    return str(tmp_path / name)

def _compare_scene(a, b):
    assert np.allclose(a["points"], b["points"])
    assert len(a["lines"]) == len(b["lines"])
    for x, y in zip(a["lines"], b["lines"]):
        assert np.allclose(x, y)
    assert len(a["areas"]) == len(b["areas"])
    for x, y in zip(a["areas"], b["areas"]):
        assert np.allclose(x["polygon"], y["polygon"])
    assert np.allclose(a["labels_init"], b["labels_init"])
    assert np.allclose(a["WH"], b["WH"])
    assert np.allclose(a["anchors"], b["anchors"])
    if "scene_script" in a or "scene_script" in b:
        assert a.get("scene_script") == b.get("scene_script")

def test_generate_scene_minimum_shape():
    data = generate_scene(canvas_size=(640,480), n_points=5, n_lines=1, n_areas=1, seed=123)
    assert data["points"].shape == (5,2)
    assert isinstance(data["lines"], list) and len(data["lines"]) == 1
    for pl in data["lines"]:
        arr = np.asarray(pl, float)
        assert arr.ndim == 2 and arr.shape[1] == 2
    assert len(data["areas"]) == 1
    n_labels = 5 + 1 + 1
    assert data["labels_init"].shape == (n_labels,2)
    assert data["WH"].shape == (n_labels,2)
    assert data["anchors"].shape == (n_labels,2)


def test_seed_reproducible():
    d1 = generate_scene(canvas_size=(640,480), n_points=6, n_lines=2, n_areas=2, seed=42)
    d2 = generate_scene(canvas_size=(640,480), n_points=6, n_lines=2, n_areas=2, seed=42)
    _compare_scene(d1, d2)


def test_cache_roundtrip(tmp_path):
    base = _tmp_cache(tmp_path)
    gen_cfg = {
        "frame": {"width": 320, "height": 240},
        "counts": {"n_points": 3, "n_lines": 1, "n_areas": 1},
    }
    key = config_hash(gen_cfg)
    expected = base.replace(".npz", f"_{key}.npz")
    d0 = get_scene(use_random=True, cache_path=base, gen_cfg=gen_cfg, seed=7)
    assert os.path.exists(expected)
    d1 = get_scene(use_random=False, cache_path=base, gen_cfg=gen_cfg)
    d2 = get_scene(use_random=False, cache_path=base, gen_cfg=gen_cfg)
    _compare_scene(d0, d1)
    _compare_scene(d1, d2)


def test_generate_to_solve_pipeline():
    """Full pipeline: generate a random scene and solve its layout."""
    scene = generate_scene(canvas_size=(480, 320), n_points=4, n_lines=1, n_areas=1, seed=0)
    L = scene["labels_init"].shape[0]
    sp = SolvePack(
        L=L,
        P0=scene["labels_init"],
        active_mask0=np.ones(L, dtype=bool),
        scene=scene,
        cfg={"compute": {"weights": {"anchor.spring": 1.0}, "eps": {"numeric": 1e-12}}},
        stages=[{"iters": 5, "solver": "lbfgs"}],
        passes=["schedule", "capture"],
    )
    view = solve(sp)
    P = view.last.P
    assert P.shape == scene["labels_init"].shape
    assert np.all(np.isfinite(P))
