import numpy as np
from cartoweave.data.api import build_solvepack_direct

def test_sequential_equals_labels():
    sp = build_solvepack_direct(
        frame_size=(100, 100),
        n_labels=5,
        steps={"kind": "sequential", "steps": 5},
        seed=0,
    )
    assert len(sp.stages) == 5
    masks = [st["mask_override"] for st in sp.stages]
    assert all(m.shape == (5,) for m in masks)
    assert masks[0].sum() == 1
    for i in range(1, 5):
        diff = masks[i] & ~masks[i - 1]
        assert diff.sum() == 1
    assert np.array_equal(sp.active_mask0, masks[0])

def test_sequential_fewer_than_labels():
    sp = build_solvepack_direct(
        frame_size=(100, 100),
        n_labels=5,
        steps={"kind": "sequential", "steps": 3},
        seed=0,
    )
    assert len(sp.stages) == 3
    m0, m1, m2 = [st["mask_override"] for st in sp.stages]
    assert m0.sum() == 1
    assert (m1 & ~m0).sum() == 1
    assert np.all(m2)
    assert (m2 & ~m1).sum() == 3

def test_no_actions_single_stage():
    sp = build_solvepack_direct(
        frame_size=(100, 100),
        n_labels=5,
        steps={"kind": "none"},
        seed=0,
    )
    assert len(sp.stages) == 1
    st = sp.stages[0]
    assert st.get("mask_override") is None
    assert st["iters"] > 0
    assert np.all(sp.active_mask0)

def test_determinism_and_contract():
    params = dict(frame_size=(80, 60), n_labels=4, steps={"kind": "none"})
    sp1 = build_solvepack_direct(**params, seed=42)
    sp2 = build_solvepack_direct(**params, seed=42)
    assert np.allclose(sp1.scene["WH"], sp2.scene["WH"])
    assert np.allclose(sp1.P0, sp2.P0)
    assert sp1.scene["WH"].shape == (sp1.L, 2)
    assert len(sp1.scene["labels"]) == sp1.L
    assert sp1.active_mask0.shape == (sp1.L,)
