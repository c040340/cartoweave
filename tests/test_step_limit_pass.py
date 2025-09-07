import numpy as np


def test_step_limit_respects_yaml():
    from cartoweave.compute.passes import StepLimitPass

    class DummyPM:
        cfg = {"passes": {"step_limit": {"max_step_norm": 0.1}}}

    p = StepLimitPass(max_step_norm=1.5)
    p.pm = DummyPM()
    wrapped = p.wrap_step(lambda a, b, m: b)
    P0 = np.zeros((3, 2))
    P1 = np.ones((3, 2)) * 10.0
    out = wrapped(P0, P1, {})
    assert np.linalg.norm(out - P0) <= 0.1000001


def test_step_limit_disable():
    from cartoweave.compute.passes import StepLimitPass

    class DummyPM:
        cfg = {"passes": {"step_limit": {"max_step_norm": 0}}}

    p = StepLimitPass(max_step_norm=1.5)
    p.pm = DummyPM()
    wrapped = p.wrap_step(lambda a, b, m: b)
    P0 = np.zeros((1, 2))
    P1 = np.ones((1, 2))
    out = wrapped(P0, P1, {})
    assert np.allclose(out, P1)

