import numpy as np
import pytest

from cartoweave.engine.calibration import auto_calibrate_k, ema_update_k
from cartoweave.config.presets import minimal_cfg
from cartoweave.api import solve_frame
import numpy as np


def tiny_scene():
    scene = {
        "frame_size": (100.0, 100.0),
        "labels_init": np.array([[10.0, 10.0], [12.0, 10.0], [50.0, 50.0]], float),
        "WH": np.array([[10.0, 5.0], [10.0, 5.0], [10.0, 5.0]], float),
        "anchors": np.array([[10.0, 10.0], [12.0, 10.0], [50.0, 50.0]], float),
        "points": np.zeros((0, 2), float),
        "lines": np.zeros((0, 4), float),
        "areas": [],
    }
    return scene


def test_auto_calibrate_scales_k():
    sc = tiny_scene()
    cfg = minimal_cfg()
    cfg.update(
        {
            "ll.k.repulse": 1.0,
            "boundary.k.wall": 1.0,
            "calib.k.enable": True,
            "calib.k.base_term": "boundary.wall",
        }
    )

    P0 = sc["labels_init"]
    k_hat = auto_calibrate_k(sc, P0, cfg)
    # ll term should be scaled, boundary skipped (base term)
    assert "ll.k.repulse" in k_hat
    assert "boundary.k.wall" not in k_hat
    s = k_hat["ll.k.repulse"] / cfg["ll.k.repulse"]
    assert 0.33 <= s <= 3.0


def test_auto_calibrate_disabled_noop():
    sc = tiny_scene()
    cfg = minimal_cfg()
    cfg.update({"ll.k.repulse": 1.0, "calib.k.enable": False})
    P0 = sc["labels_init"]
    assert auto_calibrate_k(sc, P0, cfg) == {}


def test_autocalibrate_base_fallback():
    sc = tiny_scene()
    cfg = minimal_cfg()
    cfg.update({"ll.k.repulse": 1.0, "calib.k.enable": True})
    P0 = sc["labels_init"]
    k_hat = auto_calibrate_k(sc, P0, cfg)
    assert "ll.k.repulse" in k_hat


def test_ema_blending():
    prev = {"ll.k.repulse": 1.0}
    first = ema_update_k(prev, {"ll.k.repulse": 3.0}, 0.5)
    # Between previous and new
    assert first["ll.k.repulse"] == pytest.approx(2.0)
    second = ema_update_k(first, {"ll.k.repulse": 0.5}, 0.5)
    # Monotonic towards the new value
    assert second["ll.k.repulse"] == pytest.approx(1.25)


def test_solver_keeps_k_constant():
    sc = tiny_scene()
    cfg = minimal_cfg()
    cfg.update(
        {
            "ll.k.repulse": 0.5,
            "boundary.k.wall": 1.0,
            "calib.k.enable": True,
            "calib.k.base_term": "boundary.wall",
        }
    )
    P0 = sc["labels_init"]
    k_hat = auto_calibrate_k(sc, P0, cfg)
    cfg.update(k_hat)
    k_before = cfg["ll.k.repulse"]
    solve_frame(sc, cfg, mode="lbfgs")
    assert cfg["ll.k.repulse"] == pytest.approx(k_before)
