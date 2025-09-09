import numpy as np

from cartoweave.compute.forces import get_evaluate


def test_ll_disk_active_without_rect():
    """Enabling only ``ll.disk`` should still produce forces for default labels."""

    scene = {"labels": [{"WH": [10.0, 10.0]}, {"WH": [10.0, 10.0]}]}
    P = np.array([[0.0, 0.0], [15.0, 0.0]], float)
    cfg = {"public": {"forces": {"ll": {"disk": {"enable": True}, "rect": {"enable": False}}}}}

    fn = get_evaluate("ll.disk")
    E, F, _ = fn(scene, P, {}, cfg)

    assert E > 0.0 and np.linalg.norm(F) > 0.0

