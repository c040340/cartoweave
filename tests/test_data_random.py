# tests/test_data_random.py
from __future__ import annotations
import os, shutil, json
from cartoweave.data.random import generate_scene, get_scene, make_timeline

def _tmp_cache(tmp_path, name="scene_cache.json"):
    p = tmp_path / name
    return str(p)

def test_generate_scene_minimum_shape():
    data = generate_scene(canvas_size=(640, 480), n_points=5, n_lines=1, n_areas=1, seed=123)
    assert set(data.keys()) >= {"points","lines","areas","label_mode","label_specs"}
    assert len(data["points"]) == 5
    assert len(data["lines"]) == 1
    assert len(data["areas"]) == 1
    # 每个元素都有 label_mode/specs
    ids = [*map(lambda p:p["id"], data["points"]),
           *map(lambda l:l["id"], data["lines"]),
           *map(lambda a:a["id"], data["areas"])]
    for eid in ids:
        assert eid in data["label_mode"]
        assert eid in data["label_specs"]
        s = data["label_specs"][eid]
        assert "single" in s and "detail" in s
        assert s["single"]["w"] > 0 and s["detail"]["w"] >= s["single"]["w"]

def test_seed_reproducible():
    d1 = generate_scene(canvas_size=(640, 480), n_points=6, n_lines=2, n_areas=2, seed=42)
    d2 = generate_scene(canvas_size=(640, 480), n_points=6, n_lines=2, n_areas=2, seed=42)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

def test_cache_roundtrip(tmp_path):
    cache = _tmp_cache(tmp_path)
    # 首次：use_random=True 写入
    d1 = get_scene(use_random=True, cache_path=cache, canvas_size=(320,240), n_points=3, n_lines=1, n_areas=1, seed=7)
    assert os.path.exists(cache)
    # 第二次：use_random=False 读取，应与缓存一致
    d2 = get_scene(use_random=False, cache_path=cache, canvas_size=(999,999), n_points=99, n_lines=0, n_areas=0, seed=None)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)

def test_auto_make_timeline(tmp_path):
    cache = _tmp_cache(tmp_path)
    # 先写一份无 timeline 的缓存
    _ = get_scene(use_random=True, cache_path=cache, with_timeline=False, canvas_size=(320,240), n_points=3, n_lines=1, n_areas=1, seed=1)
    # 再读：要求自动补齐 timeline（make_timeline_fn 用一个假的）
    def fake_tl(d: dict, **kw): return [{"op":"appear","id": d["points"][0]["id"], "mode":"single", "aid":0}]
    d = get_scene(use_random=False, cache_path=cache, with_timeline=True, auto_make_timeline_if_missing=True,
                  make_timeline_fn=fake_tl, timeline_kwargs={"seed": None})
    assert "timeline" in d and isinstance(d["timeline"], list) and len(d["timeline"]) >= 1


def test_timeline_seed_reproducible():
    base = generate_scene(canvas_size=(640, 480), n_points=4, n_lines=1, n_areas=1, seed=100)
    tl1 = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=555, n_extra=5)
    tl2 = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=555, n_extra=5)
    assert tl1 == tl2


def test_timeline_interleaves_change_and_appear():
    base = generate_scene(canvas_size=(640, 480), n_points=4, n_lines=1, n_areas=1, seed=0)
    tl = make_timeline(base["points"], base["lines"], base["areas"], base["label_mode"], seed=1, n_extra=5)
    first_ops: dict[str, str] = {}
    for act in tl:
        eid, op = act["id"], act["op"]
        if eid not in first_ops:
            first_ops[eid] = op
            assert op == "appear"
    last_appear_idx = max(i for i, a in enumerate(tl) if a["op"] == "appear")
    assert any(a["op"] in ("change", "hide") for a in tl[:last_appear_idx])


def test_cache_roundtrip_with_timeline(tmp_path):
    cache = _tmp_cache(tmp_path, name="scene_tl_cache.json")
    d1 = get_scene(use_random=True, cache_path=cache, canvas_size=(320,240), n_points=3, n_lines=1, n_areas=1,
                   seed=7, with_timeline=True, timeline_kwargs={"seed": 5})
    assert os.path.exists(cache)
    d2 = get_scene(use_random=False, cache_path=cache, canvas_size=(999,999), n_points=99, n_lines=0, n_areas=0,
                   with_timeline=True)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
