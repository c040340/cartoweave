import numpy as np
from cartoweave.data.sampling.poisson import poisson_disc


def test_poisson_min_distance():
    rng = np.random.default_rng(0)
    pts = poisson_disc(rng, 100.0, 100.0, 10.0)
    dists = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() >= 10.0 - 1e-6
