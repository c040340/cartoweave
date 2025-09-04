import numpy as np
from cartoweave.utils.numerics import sigmoid_np, d_sigmoid_np, softplus_np

def test_sigmoid_np_monotonic():
    x = np.linspace(-200, 200, 4001)
    s = sigmoid_np(x)
    assert np.all((s >= 0.0) & (s <= 1.0))
    d = np.diff(s)
    assert np.all(d >= 0)
    assert np.all(d[1700:2300] > 0)
    d = d_sigmoid_np(x)
    assert np.all(d >= 0)
    assert np.all(d <= 0.25 + 1e-12)
    assert abs(d_sigmoid_np(np.array([0.0]))[0] - 0.25) < 1e-3

def test_softplus_np_asymptotic():
    z = np.array([-1000.0, 0.0, 1000.0])
    sp = softplus_np(z)
    assert np.isclose(sp[0], 0.0, atol=1e-6)
    assert np.isclose(sp[1], np.log(2.0), atol=1e-6)
    assert np.isclose(sp[2], 1000.0, rtol=1e-6)
