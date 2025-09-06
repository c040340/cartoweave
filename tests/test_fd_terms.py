import numpy as np
from cartoweave.compute.eval import energy_and_grad_full


def test_fd_boundary_wall(make_min_scene, fd_grad):
    P0, labels, scene, active_mask, cfg = make_min_scene(L=1, seed=1)
    cfg["compute"]["weights"] = {"boundary.wall": 1.0}

    def E_only(P_flat):
        P = P_flat.reshape(P0.shape)
        E, _, _ = energy_and_grad_full(P, labels, scene, active_mask, cfg)
        return E

    _, g_ana, _ = energy_and_grad_full(P0, labels, scene, active_mask, cfg)
    _, g_fd = fd_grad(E_only, P0)
    np.testing.assert_allclose(g_ana, g_fd, rtol=1e-4, atol=1e-6)
