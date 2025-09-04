import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.panels import (
    draw_layout,
    normalize_comps_for_info,
    select_terms_for_arrows,
    ALL_FORCE_KEYS,
)
from cartoweave.config.presets import VIZ_FORCE_CONFIG, minimal_cfg
from cartoweave.orchestrators.scene_script_runner import run_scene_script


def test_sources_for_step_visibility():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], float)
    lines = [np.array([[0.0, 0.0], [1.0, 0.0]], float), np.array([[1.0, 1.0], [2.0, 1.0]], float)]
    areas = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], float)]
    labels = [
        {"anchor_kind": "point", "anchor_index": 0},
        {"anchor_kind": "line", "anchor_index": 1},
        {"anchor_kind": "area", "anchor_index": 0},
    ]
    rec = {"P": np.zeros((3, 2)), "comps": {}, "meta": {}}
    info = {
        "scene": {"points": points, "lines": lines, "areas": areas, "labels": labels},
        "history": {
            "records": [rec, rec],
            "scene_steps": [
                {"name": "s0", "rec_start": 0, "rec_end": 1, "active_ids_viz": [0, 1]},
                {"name": "s1", "rec_start": 1, "rec_end": 2, "active_ids_viz": [2]},
            ],
        },
    }
    payload = build_viz_payload(info)
    f0 = payload["frames"][0]["sources_for_step"]
    assert len(f0["points"]) == 1 and np.allclose(f0["points"][0], points[0])
    assert len(f0["lines"]) == 1 and np.allclose(f0["lines"][0], lines[1])
    assert f0["areas"] == []


def test_draw_layout_circle_label():
    fig, ax = plt.subplots()
    pos = np.array([[10.0, 10.0]], float)
    labels = [{"mode": "circle"}]
    wh = np.array([[5.0, 5.0]], float)
    patches = draw_layout(ax, pos, labels, wh, frame_w=20, frame_h=20)
    assert len(patches) == 1
    from matplotlib.patches import Circle

    assert isinstance(patches[0][1], Circle)
    assert len(ax.texts) == 0


def test_normalize_comps_for_info_lists_all():
    N = 2
    comps = {"ll.rect": np.ones((N, 2)), "boundary.wall": np.zeros((N, 2))}
    out = normalize_comps_for_info(comps, N)
    for k in ALL_FORCE_KEYS:
        assert k in out
        assert out[k].shape == (N, 2)
    assert np.allclose(out["pl.rect"], 0.0)


def test_select_terms_for_arrows_filtering():
    comps = {
        "strong": np.array([[1.0, 0.0], [0.5, 0.0]]),
        "weak": np.array([[1e-4, 0.0], [1e-4, 0.0]]),
    }
    cfg = dict(VIZ_FORCE_CONFIG)
    cfg["arrows_min_ratio"] = 0.5
    terms = select_terms_for_arrows(comps, cfg)
    assert terms == ["strong"]
    out = normalize_comps_for_info(comps, 2)
    assert set(comps.keys()).issubset(out.keys())


def test_no_force_leakage_assertion():
    scene = {
        "labels": [
            {"id": 0, "visible": True, "mode": "rect"},
            {"id": 1, "visible": False, "mode": "rect"},
        ],
        "labels_init": np.zeros((2, 2), float),
        "WH": np.ones((2, 2), float),
        "anchors": np.zeros((2, 2), float),
    }
    scene_script = [{"name": "s0", "op": "enter", "id": 0}]
    solve_plan = {"stages": [{"name": "st"}]}
    cfg = minimal_cfg()
    cfg.update({"ll.k.repulse": 10.0, "DEBUG_VISIBILITY_ASSERTS": True})
    run_scene_script(scene, scene_script, solve_plan, cfg)
