import numpy as np

from cartoweave.compute.array_utils import expand_subset, expand_comps_subset


def test_expand_subset_keeps_inactive_rows():
    prev = np.array([[0, 0], [1, 1], [2, 2]], float)
    active_idx = np.array([0, 2])
    sub = np.array([[10, 10], [20, 20]], float)
    out = expand_subset(prev, active_idx, sub)
    expected = np.array([[10, 10], [1, 1], [20, 20]], float)
    assert np.allclose(out, expected)


def test_expand_comps_subset_merges_terms():
    prev = {"a": np.array([[1, 1], [2, 2], [3, 3]], float)}
    active_idx = np.array([1])
    sub = {"a": np.array([[10, 10]], float), "b": np.array([[5, 5]], float)}
    out = expand_comps_subset(prev, active_idx, sub)
    assert set(out.keys()) == {"a", "b"}
    assert np.allclose(out["a"], np.array([[1, 1], [10, 10], [3, 3]], float))
    assert np.allclose(out["b"], np.array([[0, 0], [5, 5], [0, 0]], float))
