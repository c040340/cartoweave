def test_enabled_terms_phase_split():
    from cartoweave.compute.forces import enabled_terms
    cfg = {
        "compute": {
            "weights": {
                "anchor.spring": 1.0,
                "pl.rect": 1.0,
                "area.embed": 0.0,
            }
        }
    }
    pre = set(enabled_terms(cfg, phase="pre_anchor"))
    anc = set(enabled_terms(cfg, phase="anchor"))
    assert "pl.rect" in pre and "anchor.spring" not in pre
    assert "anchor.spring" in anc and "pl.rect" not in anc
    assert "area.embed" not in pre and "area.embed" not in anc  # disabled
