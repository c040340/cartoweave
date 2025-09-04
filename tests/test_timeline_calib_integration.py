import numpy as np
import pytest

from cartoweave.orchestrators.timeline import run_timeline


def dummy_solver(scene, cfg):
    # returns dummy positions & info
    return np.zeros((len(scene.get("labels", [])), 2)), {"used_cfg": dict(cfg)}


def test_timeline_no_calibration_keeps_cfg():
    base_cfg = {
        "ll.k.repulse": 1.0,
        "calib.k.enable": False,
        "calib.shape.enable": False,
    }
    schedule = [{"scene": {"labels": [1, 2]}}]
    results = run_timeline(schedule, base_cfg, dummy_solver)
    assert isinstance(results, list)
    # no recalibration → ll.k.repulse unchanged
    assert results[0][1]["used_cfg"]["ll.k.repulse"] == 1.0


def test_timeline_with_calibration(monkeypatch):
    from cartoweave.engine import calibration

    # Patch auto_calibrate_k to deterministic result
    monkeypatch.setattr(
        calibration, "auto_calibrate_k", lambda s, P0, cfg: {"ll.k.repulse": 2.0}
    )
    monkeypatch.setattr(calibration, "crowding_score", lambda s, P0, cfg: 0.9)
    monkeypatch.setattr(
        calibration, "should_recalibrate_k", lambda C, Cp, a, step, cfg: True
    )

    base_cfg = {
        "ll.k.repulse": 1.0,
        "calib.k.enable": True,
        "calib.shape.enable": False,
        "calib.k.ema_alpha": 1.0,
    }
    schedule = [{"scene": {"labels": [1, 2]}}]
    results = run_timeline(schedule, base_cfg, dummy_solver)
    cfg_used = results[0][1]["used_cfg"]
    # ll.k.repulse must have been updated to 2.0 by calibration
    assert abs(cfg_used["ll.k.repulse"] - 2.0) < 1e-9

