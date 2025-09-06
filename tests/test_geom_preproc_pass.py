import numpy as np

from cartoweave.compute.passes.manager import PassManager
from cartoweave.compute.passes.geom_preproc import GeomPreprocPass
from cartoweave.compute.eval import energy_and_grad_full
from cartoweave.contracts.solvepack import Label, Scene, SolvePack


def _dummy_energy(P, labels, scene, mask, cfg):
    return 0.0, np.zeros_like(P), {}


def make_label():
    return Label(
        id=0,
        kind="line",
        polyline=[(0.0, 0.0), (0.0, 0.0), (1e-10, 0.0), (1.0, 0.0)],
        anchor={"mode": "xy", "xy": (0.0, 0.0)},
        meta={},
    )


def test_geom_preproc_cleans_polyline_and_emits_event():
    P = np.zeros((1, 2))
    labels = [make_label()]
    mask = np.array([True])
    cfg = {}
    pm = PassManager(cfg, ["geom_preproc"])
    wrapped = pm.wrap_energy(_dummy_energy)
    wrapped(P, labels, {}, mask, cfg)
    # cleaned label cached
    gp = next(p for p in pm.passes if isinstance(p, GeomPreprocPass))
    cleaned = gp.stage_cache[0][0]
    assert cleaned.polyline == [(0.0, 0.0), (1.0, 0.0)]
    events = [e for e in pm.pop_events() if e.get("info") == "cleanup"]
    assert len(events) == 1
    m = events[0]["metrics"]
    assert m["collapsed"] == 1 and m["dropped_segments"] == 1 and m["affected_labels"] == 1


def test_energy_after_cleanup_no_shape_errors():
    P = np.zeros((1, 2))
    label = Label(
        id=0,
        kind="line",
        polyline=[(0.0, 0.0), (0.0, 0.0)],
        anchor={"mode": "xy", "xy": (0.0, 0.0)},
        meta={},
    )
    mask = np.array([True])
    cfg = {}
    pm = PassManager(cfg, ["geom_preproc"])
    wrapped = pm.wrap_energy(energy_and_grad_full)
    E, G, comps = wrapped(P, [label], Scene(frame_size=(10.0, 10.0)).model_dump(), mask, cfg)
    assert G.shape == (1, 2)
    events = [e for e in pm.pop_events() if e.get("info") == "cleanup"]
    assert events[0]["metrics"]["affected_labels"] == 1
