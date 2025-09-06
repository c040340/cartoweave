from cartoweave.compute.eval import energy_and_grad_full
import numpy as np


def test_boundary_asymmetry_is_from_far_sides(make_min_scene):
    P, labels, scene, active, cfg = make_min_scene(L=2, seed=0)
    cfg["compute"]["weights"] = {"boundary.wall": 1.0}

    W, H = scene["frame_size"]
    P[:] = np.array([[5.0, H / 2], [W - 5.0, H / 2]])

    E, g, comps = energy_and_grad_full(P, labels, scene, active, cfg)
    F = -g
    assert F[0, 0] > 0.0
    assert F[1, 0] < 0.0
    assert np.isclose(abs(F[0, 0]), abs(F[1, 0]), rtol=0.25)
