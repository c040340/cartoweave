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
import numpy as np
import pytest

@pytest.fixture
def make_min_scene():
    """
    Return a minimal, consistent tuple:
      P0: (L,2) float
      labels: list of label dicts (at least anchor_kind for each)
      scene: dict with required fields like WH, frame_size, etc.
      active_mask: (L,) bool
      cfg: dict with 'compute' keys (weights, passes) as used by the current code
    """
    def _fn(L=4, frame=(1000.0, 1000.0), W=60.0, H=24.0, seed=0):
        rng = np.random.default_rng(seed)
        P0 = rng.uniform(100, 300, size=(L, 2)).astype(float)
        labels = [{"anchor_kind": "none"} for _ in range(L)]
        WH = np.array([[W, H]] * L, float)
        active_mask = np.ones(L, dtype=bool)
        scene = dict(
            frame=0,
            frame_size=frame,
            labels_init=P0,
            WH=WH,
            labels=labels,
        )
        cfg = {
            "compute": {
                "weights": {
                    "anchor.spring": 1.0,
                    "boundary.wall": 1.0,
                    "ll.rect": 1.0,
                    "pl.rect": 1.0,
                },
                "passes": {
                    "nan_guard": {},
                    "grad_clip": {"max_norm": 1.0},
                    "capture": {"every": 1},
                },
            }
        }
        return P0, labels, scene, active_mask, cfg
    return _fn

@pytest.fixture
def fd_grad():
    """ (E(x+e)-E(x-e))/(2e) central diff for test only. """
    def _fn(E_fn, X, eps=1e-6):
        X = np.asarray(X, float)
        g = np.zeros_like(X)
        fx = E_fn(X)
        for i in range(X.size):
            xp = X.copy(); xp.flat[i] += eps
            xm = X.copy(); xm.flat[i] -= eps
            g.flat[i] = (E_fn(xp) - E_fn(xm)) / (2 * eps)
        return fx, g
    return _fn


import pytest
from cartoweave.contracts.solvepack import SolvePack, Scene
from cartoweave.compute.eval import energy_and_grad_full


@pytest.fixture
def as_legacy_pack():
    """Wrap v2 SolvePack with legacy attributes expected by solve()."""

    class LegacySolvePack:
        def __init__(self, base: SolvePack):
            self._base = base
            self.N = base.N
            self.P0 = base.P0
            self.active0 = base.active0
            self.labels0 = base.labels0
            self.scene0 = base.scene0
            self.cfg = base.cfg
            self.mode = None
            self.passes = []
            self.stages = []
            self.energy_and_grad = lambda P, labels, scene, active, cfg: energy_and_grad_full(
                P, labels, scene.to_dict() if hasattr(scene, "to_dict") else scene, active, cfg
            )

        @property
        def L(self):
            return self._base.L

        @property
        def active_mask0(self):
            return self._base.active_mask0

        def validate(self):
            return self._base.validate()

    def _wrap(sp: SolvePack):
        assert hasattr(sp, "scene0")
        assert isinstance(sp.scene0, Scene), "scene0 must be Scene for current solve()"
        return LegacySolvePack(sp)

    return _wrap
