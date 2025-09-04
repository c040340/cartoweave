import copy
from cartoweave.config.layering import apply_shape_profile
from cartoweave.config.schema import SPEC
from cartoweave.config.profiles.shapes import SHAPE_PROFILES

# Extend SPEC with keys used in shape profiles but absent in base schema
SPEC.update({
    "boundary.wall_eps": {"type": float, "range": (1e-12, 10.0), "mutable": "frozen", "group": "shape"},
    "area.cross.alpha": {"type": float, "range": (0.1, 50.0), "mutable": "frozen", "group": "shape"},
})


def _base_cfg():
    # Pretend caller loaded defaults elsewhere; we only need relevant keys.
    return {
        "ll.edge_power": 1.5,            # frozen, present
        "boundary.wall_power": 2.0,      # frozen, present
        "beta.softplus.dist": 1.2,       # frozen, present
        "anchor.spring.alpha": 7.0,      # frozen, present
        "area.cross.alpha": 7.0,         # frozen, present
        "boundary.wall_eps": 0.3,        # frozen, present (sigma scaled)
        "eps.abs": 0.25,                 # frozen, present (sigma scaled)
        "focus.k.attract": 1.0,          # not frozen, should NOT be changed by shape profile
    }


def test_profile_disabled_noop():
    cfg = _base_cfg()
    n = apply_shape_profile(cfg, name="default", enable=False, sigma_scale=1.0)
    assert n == 0
    assert cfg["ll.edge_power"] == 1.5


def test_apply_default_profile_overrides_frozen_only():
    cfg = _base_cfg()
    n = apply_shape_profile(cfg, name="default", enable=True, sigma_scale=1.0)
    assert n >= 1
    # frozen keys should be set to profile values
    assert cfg["ll.edge_power"] == 2.0
    assert cfg["boundary.wall_power"] == 3.0
    assert cfg["beta.softplus.dist"] == 2.0
    assert cfg["anchor.spring.alpha"] == 10.0
    assert cfg["area.cross.alpha"] == 10.0
    # eps keys scaled by sigma=1.0 => unchanged vs profile values
    assert abs(cfg["boundary.wall_eps"] - 0.5) < 1e-9
    assert abs(cfg["eps.abs"] - 0.5) < 1e-9
    # non-frozen key must NOT be changed
    assert cfg["focus.k.attract"] == 1.0


def test_missing_key_ignored():
    cfg = _base_cfg()
    del cfg["ll.edge_power"]
    n = apply_shape_profile(cfg, name="default", enable=True, sigma_scale=1.0)
    assert "ll.edge_power" not in cfg
    assert n == len(SHAPE_PROFILES["default"]["fixed"]) - 1


def test_apply_dense_inheritance_and_sigma_scale():
    cfg = _base_cfg()
    n = apply_shape_profile(cfg, name="dense", enable=True, sigma_scale=2.0)
    assert n >= 1
    # dense overrides from default plus its own values
    assert cfg["ll.edge_power"] == 3.0        # from dense
    assert cfg["boundary.wall_power"] == 3.0  # from dense
    assert cfg["beta.softplus.dist"] == 2.4   # from dense
    assert cfg["anchor.spring.alpha"] == 12.0
    assert cfg["area.cross.alpha"] == 12.0
    # eps keys: default profile had 0.5; dense doesn't override them, but we still scale on apply
    assert abs(cfg["boundary.wall_eps"] - 1.0) < 1e-9   # 0.5 * 2.0
    assert abs(cfg["eps.abs"] - 1.0) < 1e-9


def test_unknown_profile_raises():
    cfg = _base_cfg()
    try:
        apply_shape_profile(cfg, name="unknown_profile", enable=True, sigma_scale=1.0)
        assert False, "should have raised"
    except KeyError:
        pass
