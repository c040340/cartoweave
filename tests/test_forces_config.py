import numpy as np
import pytest
import numpy as np

from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.config.loader import load_compute_config
from cartoweave.compute.forces import term_params_map, enabled_terms


def test_force_enable_toggle():
    compute_cfg = {"public": {"forces": {"anchor": {"spring": {"enable": True, "k_local": 1.0}}}}}
    names = enabled_terms(compute_cfg, phase="pre_anchor") + enabled_terms(compute_cfg, phase="anchor")
    pmap = term_params_map(compute_cfg)
    terms = [{"name": n, "k_local": float(pmap[n]["k_local"])} for n in names]
    assert terms == [{"name": "anchor.spring", "k_local": 1.0}]


def test_scaling_by_k():
    scene = {"frame_size": (10.0, 10.0)}
    labels = ["pt"]
    P = np.array([[1.0, 1.0]], dtype=float)
    mask = np.array([True])
    cfg1 = {"public": {"forces": {"anchor": {"spring": {"enable": True, "k_local": 1.0}}}}}
    cfg2 = {"public": {"forces": {"anchor": {"spring": {"enable": True, "k_local": 2.0}}}}}
    E1, G1, _ = energy_and_grad_full(P, labels, scene, mask, cfg1)
    E2, G2, _ = energy_and_grad_full(P, labels, scene, mask, cfg2)
    assert pytest.approx(E2, rel=1e-6) == 2.0 * E1
    assert pytest.approx(float(np.linalg.norm(G2)), rel=1e-6) == 2.0 * float(np.linalg.norm(G1))


def test_zero_force_distance():
    scene = {"frame_size": (10.0, 10.0)}
    labels = ["pt"]
    P = np.array([[1.0, 0.0]], dtype=float)
    mask = np.array([True])
    cfg0 = {
        "public": {"forces": {"anchor": {"spring": {"enable": True, "k_local": 1.0, "zero_dist": 0.0}}}}
    }
    cfg1 = {
        "public": {"forces": {"anchor": {"spring": {"enable": True, "k_local": 1.0, "zero_dist": 2.0}}}}
    }
    E0, G0, _ = energy_and_grad_full(P, labels, scene, mask, cfg0)
    E1, G1, _ = energy_and_grad_full(P, labels, scene, mask, cfg1)
    assert E1 < E0
    assert float(np.linalg.norm(G1)) < float(np.linalg.norm(G0))


def test_legacy_weights_rejected():
    with pytest.raises(ValueError):
        load_compute_config(overrides={"compute": {"weights": {"foo": 1.0}}})


def test_term_params_passthrough():
    scene = {
        "areas": [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]],
        "frame_size": (5.0, 5.0),
    }
    labels = [
        {"WH": (0.5, 0.5)},
        {"WH": (0.5, 0.5), "anchor": {"kind": "area", "index": 0}},
    ]
    P = np.array([[1.9, 1.0], [0.0, 0.0]], dtype=float)
    mask = np.array([True, True])
    cfg0 = {"public": {"forces": {"area": {"softout": {"enable": True, "k_push": 1.0, "min_gap": 0.0}}}}}
    cfg1 = {"public": {"forces": {"area": {"softout": {"enable": True, "k_push": 1.0, "min_gap": 1.0}}}}}
    E0, _, _ = energy_and_grad_full(P, labels, scene, mask, cfg0)
    E1, _, _ = energy_and_grad_full(P, labels, scene, mask, cfg1)
    assert E1 < E0
