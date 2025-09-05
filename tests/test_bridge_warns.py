import warnings

def test_bridge_emits_deprecation_warning():
    from cartoweave.config.bridge import translate_legacy_keys
    cfg = {"solver": {"terms": {"weights": {"pl.rect": 1.0}}, "eps": {"numeric": 1e-10}}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        out = translate_legacy_keys(cfg)
        assert out["compute"]["_bridge"]["hit_count"] >= 1
        assert any(isinstance(x.message, DeprecationWarning) for x in w)
