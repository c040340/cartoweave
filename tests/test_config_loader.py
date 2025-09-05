import yaml
import pytest
from cartoweave.config.loader import load_configs, print_effective_config


def test_load_order_and_overrides(tmp_path):
    # create layer files with same key to ensure later layers win
    internals = tmp_path / "solver.internals.yaml"
    tuning = tmp_path / "solver.tuning.yaml"
    public = tmp_path / "solver.public.yaml"
    viz = tmp_path / "viz.yaml"
    internals.write_text("internals: {}\n")
    tuning.write_text("tuning: {threshold: {abs: 1}}\n")
    public.write_text("public: {}\n")
    viz.write_text("panels: {layout: true}\n")

    cfg = load_configs(
        internals_path=str(internals),
        tuning_path=str(tuning),
        public_path=str(public),
        viz_path=str(viz),
        overrides={"solver": {"tuning": {"threshold": {"abs": 4}}}},
    )
    assert cfg["solver"]["tuning"]["threshold"]["abs"] == 4


def test_viz_separation(tmp_path):
    internals = tmp_path / "solver.internals.yaml"
    viz = tmp_path / "viz.yaml"
    internals.write_text("internals: {eps: {div: 1}}\n")
    viz.write_text("foo: 2\n")
    cfg = load_configs(
        internals_path=str(internals),
        viz_path=str(viz),
    )
    assert cfg["solver"]["internals"]["eps"]["div"] == 1
    assert cfg["viz"]["foo"] == 2


def test_print_effective_config(tmp_path):
    cfg = load_configs()
    out = tmp_path / "snap.yaml"
    print_effective_config(out)
    data = yaml.safe_load(out.read_text())
    assert "solver" in data and "viz" in data


def test_deprecated_keys_migrated(recwarn):
    cfg = load_configs(overrides={"warmup": {"steps": 5}, "solver": {"mode": "simple"}})
    assert cfg["solver"]["tuning"]["warmup"]["steps"] == 5
    assert cfg["solver"]["public"]["mode"] == "simple"
    assert "warmup" not in cfg
    assert "mode" not in cfg["solver"]
    messages = {str(w.message) for w in recwarn}
    assert any("warmup.steps" in m for m in messages)
    assert any("solver.mode" in m for m in messages)
    assert len(messages) == 2


def test_orphan_key_error():
    with pytest.raises(KeyError):
        load_configs(overrides={"typo": 1})


def test_profile_preset_respected():
    cfg = load_configs(overrides={"solver": {"public": {"profile": "fast"}}})
    assert cfg["solver"]["tuning"]["warmup"]["steps"] == 1

    cfg = load_configs(
        overrides={
            "solver": {
                "public": {"profile": "fast"},
                "tuning": {"warmup": {"steps": 7}},
            }
        }
    )
    assert cfg["solver"]["tuning"]["warmup"]["steps"] == 7


def test_force_shaping_defaults_present():
    cfg = load_configs()
    tuning = cfg["solver"]["tuning"]
    assert tuning["merge"]["mode"] == "sum"
    assert tuning["normalize"]["kind"] == "l2"
    assert tuning["topk"]["min_share"] == 0.0


def test_schema_validation_error():
    with pytest.raises(ValueError) as exc:
        load_configs(overrides={"solver": {"tuning": {"topk": {"min_share": 2.0}}}})
    assert "solver.tuning.topk.min_share" in str(exc.value)
