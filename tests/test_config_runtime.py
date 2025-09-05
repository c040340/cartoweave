import numpy as np
import pytest

from cartoweave.config.loader import load_configs
from cartoweave.config.schema import validate_config
from cartoweave.orchestrators.solve_plan import run_solve_plan


def _mini_scene():
    return dict(
        frame=0,
        frame_size=(10, 10),
        labels_init=np.zeros((0, 2), float),
        labels=[],
        WH=np.zeros((0, 2), float),
        points=[],
        lines=[],
        areas=[],
        anchors=np.zeros((0, 2), float),
    )


def test_canonical_config_and_logging(monkeypatch, caplog):
    bundle = load_configs()
    cfg = bundle.model_dump(exclude_unset=False, exclude_defaults=False)
    tuning = cfg["solver"]["tuning"]
    assert "terms" in tuning and tuning["terms"]
    assert tuning.get("anchor", {}).get("k_spring") is not None

    scene = _mini_scene()
    monkeypatch.setenv("CFG_DEBUG_FORCES", "1")
    with caplog.at_level("INFO"):
        run_solve_plan(scene, cfg, [{"name": "main"}])
    assert any("[cfg] terms_weight_sum" in r.message for r in caplog.records)


def test_legacy_term_weights_rejected():
    bad_cfg = {
        "solver": {"tuning": {"term_weights": {"boundary.k.wall": 1.0}}},
        "viz": {},
    }
    with pytest.raises(ValueError) as ei:
        validate_config(bad_cfg)
    assert "term_weights" in str(ei.value)

