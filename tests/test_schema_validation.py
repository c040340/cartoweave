import pytest
from cartoweave.config.layering import validate_cfg, snapshot, ConfigTypeRangeError, ConfigMutabilityError


def test_type_and_range_pass():
    cfg = {
        "ll.edge_power": 2.0,
        "beta.softplus.dist": 2.5,
        "focus.k.attract": 1.0,
        "calib.k.clip_q": 0.995,
        "logging.level": "INFO",
    }
    validate_cfg(cfg, "load")  # should not raise


def test_type_error():
    cfg = {"ll.edge_power": "two"}  # wrong type
    with pytest.raises(ConfigTypeRangeError):
        validate_cfg(cfg, "load")


def test_range_error():
    cfg = {"calib.k.clip_q": 0.2}  # out of [0.5,1.0]
    with pytest.raises(ConfigTypeRangeError):
        validate_cfg(cfg, "load")


def test_mutability_frozen_blocked_at_action_begin():
    cfg = {"ll.edge_power": 2.0, "focus.k.attract": 1.0}
    validate_cfg(cfg, "load"); snapshot(cfg, "_snapshot_load")
    # attempt to change frozen key at action_begin
    cfg["ll.edge_power"] = 3.0
    with pytest.raises(ConfigMutabilityError):
        validate_cfg(cfg, "action_begin")


def test_mutability_action_allowed_at_action_begin():
    cfg = {"ll.edge_power": 2.0, "focus.k.attract": 1.0}
    validate_cfg(cfg, "load"); snapshot(cfg, "_snapshot_load")
    cfg["focus.k.attract"] = 2.5  # action key can change
    validate_cfg(cfg, "action_begin")  # no raise


def test_runtime_only_changes():
    cfg = {"logging.level": "INFO", "focus.k.attract": 1.0}
    validate_cfg(cfg, "load"); snapshot(cfg, "_snapshot_load")
    validate_cfg(cfg, "action_begin"); snapshot(cfg, "_snapshot_action")
    # runtime: only runtime keys can change
    cfg["logging.level"] = "DEBUG"  # allowed
    validate_cfg(cfg, "runtime")
    # try changing action key at runtime
    cfg["focus.k.attract"] = 9.0
    with pytest.raises(ConfigMutabilityError):
        validate_cfg(cfg, "runtime")
