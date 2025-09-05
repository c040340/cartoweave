import numpy as np
from cartoweave.data.sampling.distributions import sample_split_normal_trunc


def test_split_normal_bounds_and_mean():
    rng = np.random.default_rng(1)
    arr = sample_split_normal_trunc(rng, mean=0.0, sigma_left=1.0, sigma_right=2.0, low=-3.0, high=5.0, size=1000)
    assert np.all(arr >= -3.0) and np.all(arr <= 5.0)
    m = arr.mean()
    assert -3.0 < m < 5.0
