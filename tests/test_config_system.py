import numpy as np
from pathlib import Path

import yaml  # our minimal yaml module

from cartoweave.config_loader import load_configs
from cartoweave.engine.config_utils import get_enabled_terms, get_focus_params
from cartoweave.engine.step_control import apply_step_governor


def write_yaml(path: Path, text: str):
    path.write_text(text)


def test_unknown_key_rejection(tmp_path: Path):
    cfg_text = Path("configs/config.yaml").read_text() + "unknown: 5\n"
    cfg_path = tmp_path / "config.yaml"
    write_yaml(cfg_path, cfg_text)
    viz_path = Path("configs/viz.yaml")
    try:
        load_configs(config_path=str(cfg_path), viz_path=str(viz_path))
    except KeyError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError("Unknown key did not raise")


def test_run_override_shallow(tmp_path: Path):
    cfg_path = Path("configs/config.yaml")
    viz_path = Path("configs/viz.yaml")
    run_text = """
solver:
  tuning:
    stopping:
      gtol_abs: 5e-3
viz:
  enable: false
"""
    run_path = tmp_path / "run.yaml"
    write_yaml(run_path, run_text)
    bundle = load_configs(config_path=str(cfg_path), viz_path=str(viz_path), run_path=str(run_path))
    assert bundle.core.solver.tuning.stopping.gtol_abs == 5e-3
    assert bundle.viz.enable is False


def test_disabling_boundary(tmp_path: Path):
    run_text = """
physics:
  tuning:
    enabled:
      boundary: false
"""
    run_path = tmp_path / "run.yaml"
    write_yaml(run_path, run_text)
    bundle = load_configs(run_path=str(run_path))
    enabled = get_enabled_terms(bundle.core)
    assert "boundary" not in enabled


def test_focus_center_mapping():
    bundle = load_configs()
    focus = get_focus_params(bundle.core, frame_size=(640, 480))
    assert focus.center_px == (320.0, 240.0)


def test_step_governor_caps():
    prev = np.zeros((1, 2))
    proposed = np.array([[100.0, 0.0]])
    result = apply_step_governor(prev, proposed, max_step=10.0)
    assert np.allclose(result, [[10.0, 0.0]])

