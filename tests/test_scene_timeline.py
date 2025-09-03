from __future__ import annotations
import json, os
from cartoweave.data.random import generate_scene, get_scene, make_timeline


def _cache_path(tmp_path):
    return str(tmp_path / "scene_cache.json")


def test_scene_seed_reproducible():
    d1 = generate_scene(canvas_size=(640, 480), n_points=4, n_lines=1, n_areas=1, seed=42)
    d2 = generate_scene(canvas_size=(640, 480), n_points=4, n_lines=1, n_areas=1, seed=42)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_timeline_seed_reproducible():
    base = generate_scene(canvas_size=(640, 480), n_points=4, n_lines=1, n_areas=1, seed=0)
    tl1 = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=5, n_extra=5)
    tl2 = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=5, n_extra=5)
    assert tl1 == tl2


def test_timeline_interleaves_change_and_appear():
    base = generate_scene(canvas_size=(640, 480), n_points=3, n_lines=1, n_areas=1, seed=1)
    tl = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=2, n_extra=5)
    first_ops = {}
    for act in tl:
        eid, op = act["id"], act["op"]
        if eid not in first_ops:
            first_ops[eid] = op
            assert op == "appear"
    last_appear = max(i for i, a in enumerate(tl) if a["op"] == "appear")
    assert any(a["op"] in ("change", "hide") for a in tl[:last_appear])


def test_cache_roundtrip_with_timeline(tmp_path):
    cache = _cache_path(tmp_path)
    d1 = get_scene(use_random=True, cache_path=cache, canvas_size=(320, 240), n_points=2, n_lines=1, n_areas=1,
                   seed=7, with_timeline=True, timeline_kwargs={"seed": 5})
    assert os.path.exists(cache)
    d2 = get_scene(use_random=False, cache_path=cache, canvas_size=(999, 999), n_points=0, n_lines=0, n_areas=0,
                   with_timeline=True)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
