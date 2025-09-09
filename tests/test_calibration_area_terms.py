import numpy as np
from cartoweave.compute.passes import calibration


def test_auto_calibrate_includes_area_terms(monkeypatch):
    def fake_enabled_terms(cfg, phase):
        return ["focus.attract", "area.embed"] if phase == "pre_anchor" else []

    monkeypatch.setattr(calibration, "enabled_terms", fake_enabled_terms)

    def fake_eval_force(name, scene, P, cfg, phase):
        if name == "focus.attract":
            F = np.ones((4, 2))
        else:  # area.embed
            F = np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        return 0.0, F, {}

    monkeypatch.setattr(calibration, "_eval_force", fake_eval_force)

    P0 = np.zeros((4, 2))
    cfg = {}
    calib_cfg = {"enable": True}
    ctx = {}

    scales = calibration.auto_calibrate_k({}, P0, cfg, calib_cfg, ctx)
    assert "area.embed" in scales and scales["area.embed"] > 1.0
