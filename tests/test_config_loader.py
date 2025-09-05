import yaml
import pytest

from cartoweave.config.loader import load_configs, print_effective_config


def test_load_overrides_and_viz(tmp_path):
    internals = tmp_path / "solver.internals.yaml"
    tuning = tmp_path / "solver.tuning.yaml"
    public = tmp_path / "solver.public.yaml"
    viz = tmp_path / "viz.yaml"

    internals.write_text("solver: {internals: {eps: {div: 1}}}\n")
    tuning.write_text("solver: {threshold: {abs: 1}}\n")
    public.write_text("solver: {use_retry: true}\n")
    viz.write_text("panels: {layout: true}\n")

    cfg = load_configs(
        internals_path=str(internals),
        tuning_path=str(tuning),
        public_path=str(public),
        viz_path=str(viz),
        overrides={"solver": {"threshold": {"abs": 4}}},
    )
    assert cfg["solver"]["threshold"]["abs"] == 4
    assert cfg["viz"]["panels"]["layout"] is True


def test_print_effective_config(tmp_path):
    cfg = load_configs()
    out = tmp_path / "snap.yaml"
    print_effective_config(cfg, out)
    data = yaml.safe_load(out.read_text())
    assert "solver" in data and "viz" in data


def test_legacy_key_error(tmp_path):
    tuning = tmp_path / "solver.tuning.yaml"
    tuning.write_text("term_weights: {ll.k.repulse: 1.0}\n")
    with pytest.raises(ValueError):
        load_configs(tuning_path=str(tuning))


def test_profile_preset_respected():
    cfg = load_configs(overrides={"solver": {"profile": "fast"}})
    assert cfg["solver"]["warmup"]["steps"] == 1

    cfg = load_configs(overrides={"solver": {"profile": "fast", "warmup": {"steps": 7}}})
    assert cfg["solver"]["warmup"]["steps"] == 7


def test_force_shaping_defaults_present():
    cfg = load_configs()
    solver = cfg["solver"]
    assert solver["merge"]["mode"] == "sum"
    assert solver["normalize"]["kind"] == "l2"
    assert solver["topk"]["min_share"] == 0.0


def test_schema_validation_error():
    with pytest.raises(ValueError) as exc:
        load_configs(overrides={"solver": {"topk": {"min_share": 2.0}}})
    assert "solver.topk.min_share" in str(exc.value)

