import numpy as np

from cartoweave.engine.calibration import auto_calibrate_k, ema_update_k
from cartoweave.engine.calibration import (
    apply_shape_profile_from_cfg,
)  # smoke import
from cartoweave.engine.forces import REGISTRY


class DummyTerm:
    """Simple callable returning deterministic force magnitudes."""

    def __init__(self, mags):
        self.mags = np.array(mags, dtype=float)

    def __call__(self, scene, P0, cfg, phase=None):
        n = len(self.mags)
        F = np.zeros((n, 2), float)
        F[:, 0] = self.mags
        return 0.0, F, {}


def test_auto_calibrate_scales_toward_target(tmp_path, monkeypatch):
    # install dummy terms
    REGISTRY["ll.rect"] = DummyTerm([0.1, 0.2, 0.3])
    REGISTRY["focus.attract"] = DummyTerm([0.05, 0.06, 0.07])

    def enabled_terms_stub(cfg, phase=None):
        return ["ll.rect", "focus.attract"]

    monkeypatch.setattr(
        "cartoweave.engine.calibration.enabled_terms", enabled_terms_stub
    )

    scene = {"labels": [0, 1, 2]}
    P0 = np.zeros((3, 2), float)

    cfg = {
        "ll.k.repulse": 1.0,
        "focus.k.attract": 1.0,
        "calib.k.enable": True,
        "calib.k": {
            "clip_q": 0.995,
            "p_q": 0.95,
            "min_act": 0.0,
            "clamp_min": 0.5,
            "clamp_max": 10.0,
        },
        "calib.k.target_rel": {"focus.attract": 1.0, "ll.rect": 5.0},
        "calib.k.base_term": "focus.attract",
        "calib.k.hysteresis": 0.0,
    }

    k_hat = auto_calibrate_k(scene, P0, cfg)
    assert "ll.k.repulse" in k_hat
    assert k_hat["ll.k.repulse"] > cfg["ll.k.repulse"]


def test_hysteresis_and_clamp(monkeypatch):
    REGISTRY["ll.rect"] = DummyTerm([1.0] * 10)
    REGISTRY["focus.attract"] = DummyTerm([1.0] * 10)

    def enabled_terms_stub(cfg, phase=None):
        return ["ll.rect", "focus.attract"]

    monkeypatch.setattr(
        "cartoweave.engine.calibration.enabled_terms", enabled_terms_stub
    )

    scene = {"labels": [0] * 10}
    P0 = np.zeros((10, 2), float)

    cfg = {
        "ll.k.repulse": 2.0,
        "focus.k.attract": 1.0,
        "calib.k.enable": True,
        "calib.k": {
            "clip_q": 0.995,
            "p_q": 0.95,
            "min_act": 0.0,
            "clamp_min": 0.1,
            "clamp_max": 0.2,
        },
        "calib.k.target_rel": {"focus.attract": 1.0, "ll.rect": 0.05},
        "calib.k.base_term": "focus.attract",
        "calib.k.hysteresis": 0.0,
    }

    k_hat = auto_calibrate_k(scene, P0, cfg)
    assert abs(k_hat["ll.k.repulse"] - 2.0 * 0.1) < 1e-9


def test_ema_update():
    prev = {"ll.k.repulse": 10.0}
    new = {"ll.k.repulse": 20.0, "boundary.k.wall": 3.0}
    out = ema_update_k(prev, new, alpha=0.5)
    assert abs(out["ll.k.repulse"] - 15.0) < 1e-9
    assert abs(out["boundary.k.wall"] - 3.0) < 1e-9

