import pytest

from cartoweave.config.loader import load_compute_config


def test_merge_files():
    cfg = load_compute_config()
    comp = cfg["compute"]
    assert comp["solver"]["public"]["mode"] == "lbfgsb"
    assert comp["passes"]["capture"]["every"] == 1
    assert comp["solver"]["tuning"]["lbfgsb"]["lbfgs_maxiter"] == 400


def test_legacy_keys_rejected():
    with pytest.raises(ValueError):
        load_compute_config(overrides={"solver": {"foo": 1}})
    with pytest.raises(ValueError):
        load_compute_config(overrides={"terms": {}})


def test_top_level_extra_key_rejected():
    with pytest.raises(ValueError):
        load_compute_config(overrides={"compute": {}, "foo": {}})
