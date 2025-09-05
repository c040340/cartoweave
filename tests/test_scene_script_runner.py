import numpy as np
import pytest
from cartoweave.orchestrators.scene_script_runner import (
    apply_step_to_scene,
    run_scene_script,
)
from cartoweave.api import solve_scene_script


def _scene():
    return {
        "frame_size": (100.0, 100.0),
        "labels_init": np.array([[10.0, 10.0]], float),
        "labels": [
            {
                "id": "l0",
                "visible": False,
                "mode": "single",
                "modes": {
                    "single": {"w": 80.0, "h": 20.0},
                    "detail": {"w": 120.0, "h": 20.0},
                },
            }
        ],
        "WH": np.array([[80.0, 20.0]], float),
        "points": [],
        "lines": [],
        "areas": [],
    }


def test_apply_step_to_scene_sequence():
    scene = _scene()

    apply_step_to_scene(scene, {"op": "enter", "id": "l0"})
    assert scene["labels"][0]["visible"] is True

    apply_step_to_scene(scene, {"op": "mode", "id": "l0", "mode": "detail"})
    assert scene["labels"][0]["mode"] == "detail"
    assert np.allclose(scene["WH"][0], [120.0, 20.0])

    apply_step_to_scene(scene, {"op": "disappear", "id": "l0"})
    assert scene["labels"][0]["visible"] is False


def test_run_scene_script_records_segmented():
    scene = _scene()
    scene_script = [
        {"name": "enter", "op": "enter", "id": "l0"},
        {"name": "hide", "op": "disappear", "id": "l0"},
    ]
    solve_plan = {"stages": [{"name": "stage0"}]}
    cfg = {"boundary.k.wall": 1.0}

    info = run_scene_script(scene, scene_script, solve_plan, cfg)
    hist = info.get("history", {})
    recs = hist.get("records", [])
    steps = hist.get("scene_steps", [])

    assert len(steps) == len(scene_script)
    assert steps[0]["rec_start"] == 0
    assert steps[0]["rec_end"] == steps[1]["rec_start"]
    assert steps[-1]["rec_end"] == len(recs)
    assert all("step_id" in r.get("meta", {}) for r in recs)


def test_run_scene_script_restores_full_arrays():
    scene = {
        "frame_size": (100.0, 100.0),
        "labels_init": np.array([[0.0, 0.0], [10.0, 10.0]], float),
        "labels": [
            {
                "id": "l0",
                "visible": False,
                "mode": "single",
                "modes": {"single": {"w": 10.0, "h": 10.0}},
            },
            {
                "id": "l1",
                "visible": False,
                "mode": "single",
                "modes": {"single": {"w": 10.0, "h": 10.0}},
            },
        ],
        "WH": np.array([[10.0, 10.0], [10.0, 10.0]], float),
        "points": [],
        "lines": [],
        "areas": [],
    }
    scene_script = [
        {"name": "enter0", "op": "enter", "id": "l0"},
        {"name": "enter1", "op": "enter", "id": "l1"},
        {"name": "hide1", "op": "disappear", "id": "l1"},
    ]
    plan = {"stages": [{"name": "stage0"}]}
    cfg = {"boundary.k.wall": 1.0}
    info = run_scene_script(scene, scene_script, plan, cfg)

    assert scene["WH"].shape[0] == len(scene["labels"]) == 2
    assert scene["labels_init"].shape[0] == len(scene["labels"]) == 2
    assert info.get("P_final").shape[0] == len(scene["labels"])


def test_solve_scene_script_api_type_error():
    scene = _scene()
    with pytest.raises(TypeError):
        solve_scene_script(scene, {"steps": "oops"}, {}, solve_plan={"stages": []})


def test_solve_scene_script_api_runs():
    scene = _scene()
    script = {"steps": [{"name": "enter", "op": "enter", "id": "l0"}]}
    plan = {"stages": [{"name": "stage0"}]}
    info = solve_scene_script(scene, script, {"boundary.k.wall": 1.0}, solve_plan=plan)
    hist = info.get("history", {})
    assert len(hist.get("records", [])) > 0
