import pytest

from cartoweave.config.layering import load_base_cfg, apply_calib_profile, validate_cfg


def test_apply_default_profile_fill_only():
    cfg = load_base_cfg()
    # simulate user overriding a couple of knobs beforehand
    cfg["calib.k"] = {"clip_q": 0.99}  # user choice should be preserved
    patched = apply_calib_profile(cfg, "default", fill_only=True)
    assert patched > 0

    # Expect nested dicts present
    assert isinstance(cfg.get("calib.k"), dict)
    assert isinstance(cfg.get("calib.trigger"), dict)
    assert isinstance(cfg.get("calib.k.target_rel"), dict)

    # User-provided value is preserved
    assert cfg["calib.k"]["clip_q"] == 0.99
    # Filled others exist
    assert "p_q" in cfg["calib.k"]
    assert "tau_on" in cfg["calib.trigger"]
    assert "area.cross" in cfg["calib.k.target_rel"]


def test_switch_profile_and_fill_missing():
    cfg = load_base_cfg()
    # First fill sparse
    apply_calib_profile(cfg, "sparse", fill_only=True)
    # Then fill dense; since fill_only, nothing previously filled should be overwritten
    before = cfg["calib.k.target_rel"]["area.cross"]
    apply_calib_profile(cfg, "dense", fill_only=True)
    after = cfg["calib.k.target_rel"]["area.cross"]
    assert before == after  # still sparse’s value, because fill_only=True


def test_unknown_profile_raises():
    cfg = load_base_cfg()
    with pytest.raises(KeyError):
        apply_calib_profile(cfg, "not_a_profile", fill_only=True)


def test_validate_after_profile():
    cfg = load_base_cfg()
    apply_calib_profile(cfg, "default", fill_only=True)
    # Should still validate at load (types/ranges)
    validate_cfg(cfg, phase="load")
