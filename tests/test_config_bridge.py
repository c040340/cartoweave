def test_bridge_populates_compute_namespace():
    from cartoweave.config.bridge import translate_legacy_keys

    cfg = {
        "solver": {
            "terms": {"weights": {"pl.rect": 1.0}},
            "eps": {"numeric": 1e-10},
        }
    }
    out = translate_legacy_keys(cfg)
    assert out["compute"]["weights"]["pl.rect"] == 1.0
    assert out["compute"]["eps"]["numeric"] == 1e-10
    p = out["compute"]["passes"]
    assert "grad_clip" in p and "nan_guard" in p and "capture" in p


def test_enabled_terms_uses_compute_weights():
    from cartoweave.compute.forces import enabled_terms

    cfg = {"compute": {"weights": {"anchor.spring": 1.0, "pl.rect": 0.0}}}
    assert "anchor.spring" in enabled_terms(cfg, phase="anchor")
    assert "pl.rect" not in enabled_terms(cfg, phase="pre_anchor")
