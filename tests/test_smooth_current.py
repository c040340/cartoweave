import numpy as np
from cartoweave.compute.eval import energy_and_grad_full


def test_grad_descent_decreases_energy(make_min_scene):
    P, labels, scene, active, cfg = make_min_scene(L=16, seed=0)
    E0, g, _ = energy_and_grad_full(P, labels, scene, active, cfg)
    P1 = P - 1e-3 * g
    E1, _, _ = energy_and_grad_full(P1, labels, scene, active, cfg)
    assert E1 < E0, f"Energy did not decrease under a small negative-gradient step: {E0} -> {E1}"


def test_component_sum_matches_gradient(make_min_scene):
    P, labels, scene, active, cfg = make_min_scene(L=24, seed=1)
    E, g, comps = energy_and_grad_full(P, labels, scene, active, cfg)
    Gsum = np.zeros_like(g)
    for v in comps.values():
        Gsum += v
    np.testing.assert_allclose(g, -Gsum, rtol=1e-6, atol=1e-6)

