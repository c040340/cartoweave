import numpy as np
from cartoweave.contracts.solvepack import LabelState
import pytest
from cartoweave.contracts.solvepack import SolvePack, Scene

def _scene_ok(N=2):
    return Scene(
        points=np.zeros((0, 2), float),
        lines=[],
        areas=[],
        frame_size=(800.0, 600.0),
        labels=[{"anchor_kind": "none"} for _ in range(N)],
        WH=np.ones((N, 2), float),
    )

def test_labels_length_mismatch():
    sc = _scene_ok(2)
    sp = SolvePack(N=2, P0=np.zeros((2, 2)), active0=np.ones(2, bool), labels0=[LabelState(kind='none', WH=np.ones(2))], scene0=sc, cfg={'behaviors': []})
    with pytest.raises(ValueError):
        sp.validate()

def test_WH_shape_mismatch():
    sc = _scene_ok(2)
    sc.WH = np.ones((3, 2), float)
    sp = SolvePack(N=2, P0=np.zeros((2, 2)), active0=np.ones(2, bool), labels0=[LabelState(kind='none', WH=np.ones(2)), LabelState(kind='none', WH=np.ones(2))], scene0=sc, cfg={'behaviors': []})
    with pytest.raises(ValueError):
        sp.validate()

def test_active0_int_dtype_ok():
    sc = _scene_ok(1)
    sp = SolvePack(N=1, P0=np.zeros((1, 2)), active0=np.ones(1, int), labels0=[LabelState(kind='none', WH=np.ones(2))], scene0=sc, cfg={'behaviors': []})
    sp.validate()

def test_p0_outside_frame_allowed():
    sc = _scene_ok(1)
    P0 = np.array([[5000.0, -200.0]], float)
    sp = SolvePack(N=1, P0=P0, active0=np.ones(1, bool), labels0=[LabelState(kind='none', WH=np.ones(2))], scene0=sc, cfg={'behaviors': []})
    sp.validate()
    assert sp.P0.shape == (1, 2)
