import yaml
import pytest

from cartoweave.config.loader import load_configs, print_effective_config


def test_load_overrides_and_viz(tmp_path):
    internals = tmp_path / "solver.internals.yaml"
    tuning = tmp_path / "solver.tuning.yaml"
    public = tmp_path / "solver.public.yaml"
    viz = tmp_path / "viz.yaml"

    internals.write_text("eps: {div: 1}\n")
    tuning.write_text("threshold: {abs: 1}\n")
    public.write_text("use_retry: true\n")
    viz.write_text("panels: {layout: true}\n")

    bundle = load_configs(
        internals_path=str(internals),
        tuning_path=str(tuning),
        public_path=str(public),
        viz_path=str(viz),
        overrides={"solver": {"tuning": {"threshold": {"abs": 4}}}},
    )

    assert bundle.solver.tuning.threshold.abs == 4
    assert bundle.viz.panels.layout is True


def test_print_effective_config(tmp_path):
    bundle = load_configs()
    out = tmp_path / "snap.yaml"
    print_effective_config(bundle, out)
    data = yaml.safe_load(out.read_text())
    assert "solver" in data and "viz" in data


def test_legacy_key_error(tmp_path):
    tuning = tmp_path / "solver.tuning.yaml"
    tuning.write_text("term_weights: {ll.k.repulse: 1.0}\n")
    with pytest.raises(ValueError):
        load_configs(tuning_path=str(tuning))


def test_profile_preset_respected():
    bundle = load_configs(overrides={"solver": {"public": {"profile": "fast"}}})
    assert bundle.solver.tuning.warmup.steps == 1

    bundle = load_configs(
        overrides={
            "solver": {
                "public": {"profile": "fast"},
                "tuning": {"warmup": {"steps": 7}},
            }
        }
    )
    assert bundle.solver.tuning.warmup.steps == 7


def test_force_shaping_defaults_present():
    bundle = load_configs()
    tuning = bundle.solver.tuning
    assert tuning.merge.mode == "sum"
    assert tuning.normalize.kind == "l2"
    assert tuning.topk.min_share == 0.0


def test_schema_validation_error():
    with pytest.raises(ValueError) as exc:
        load_configs(overrides={"solver": {"tuning": {"topk": {"min_share": 2.0}}}})
    assert "solver.tuning.topk.min_share" in str(exc.value)

