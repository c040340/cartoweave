import numpy as np
import pytest

@pytest.fixture
def L(): return 12

@pytest.fixture
def rng(): return np.random.default_rng(0)

@pytest.fixture
def scene(L):
    return {
        "labels": [{"mode": "rect", "anchor_kind": "none", "anchor_index": -1}] * L,
        "WH": np.ones((L, 2)),
        "frame_size": (1920, 1080),
        "points": np.zeros((0, 2)),
        "lines": [],
        "areas": [],
    }

@pytest.fixture
def mask(L): return np.ones((L,), dtype=bool)

@pytest.fixture
def P0(L): return np.zeros((L,2), dtype=float)

def assert_frame_invariants(f):
    L = f.P.shape[0]
    assert f.P.shape==(L,2) and f.G.shape==(L,2) and f.mask.shape==(L,)
    for v in f.comps.values(): assert v.shape==(L,2)

def assert_viewpack_invariants(vp):
    assert len(vp.frames)>0 and vp.last is vp.frames[-1]
    for f in vp.frames: assert_frame_invariants(f)
