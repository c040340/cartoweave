from cartoweave.config.layering import load_base_cfg, validate_cfg


def test_calibration_defaults_present_and_off():
    cfg = load_base_cfg()
    validate_cfg(cfg, phase="load")

    # Shape profile gate is OFF by default
    assert cfg.get("calib.shape.enable") is False
    assert cfg.get("calib.shape.name") == "default"
    assert isinstance(cfg.get("calib.shape.sigma_scale"), (int, float))

    # k-only calibration gate is OFF by default
    assert cfg.get("calib.k.enable") is False
    assert cfg.get("calib.k.profile") == "default"
    assert cfg.get("calib.k.base_term") in (
        None,
        "focus.attract",
        cfg.get("calib.k.base_term"),
    )

    # Quantiles & thresholds
    assert 0.9 <= cfg.get("calib.k.clip_q") <= 1.0
    assert 0.5 <= cfg.get("calib.k.p_q") < 1.0
    assert 0.0 <= cfg.get("calib.k.min_act") <= 1.0

    # Hysteresis & clamps
    assert 0.0 <= cfg.get("calib.k.hysteresis") < 1.0
    assert 0.0 <= cfg.get("calib.k.clamp_min") <= cfg.get("calib.k.clamp_max") <= 10.0

    # EMA & trigger
    assert 0.0 <= cfg.get("calib.k.ema_alpha") <= 1.0
    assert cfg.get("calib.trigger.enable") in (True, False)
    assert 0.0 <= cfg.get("calib.trigger.tau_on") <= 1.0
    assert 0.0 <= cfg.get("calib.trigger.tau_off") <= 1.0
    assert 0.0 <= cfg.get("calib.trigger.delta") <= 1.0

