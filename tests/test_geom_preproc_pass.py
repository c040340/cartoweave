import numpy as np

from cartoweave.compute.passes import PassManager
from cartoweave.compute.passes.geom_preproc import GeomPreprocPass
from cartoweave.compute.eval import energy_and_grad_full

# ---------------------------------------------------------------------------
# Import contract dataclasses.  ``AnchorSpec`` only exists in the newer
# SolvePack contract.  For legacy environments using the old ``Anchor`` model
# we fall back to that type so the tests remain backwards compatible.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only in environments with AnchorSpec
    from cartoweave.contracts.solvepack import Scene, Label, AnchorSpec
    _ANCHOR_USES_TARGET = False
except Exception:  # pragma: no cover - legacy Pydantic contract
    from cartoweave.contracts.solvepack import Scene, Label, Anchor as AnchorSpec
    _ANCHOR_USES_TARGET = True


# ---------------------------------------------------------------------------
# Helper anchor factories for readability.  ``mode`` arguments are kept for
# semantic parity with earlier APIs; the newer ``AnchorSpec`` only requires the
# ``kind``/``index`` pair.  When running against the legacy ``Anchor`` model we
# supply the necessary ``target``/``mode`` fields.
# ---------------------------------------------------------------------------


def line_anchor(index: int = 0, mode: str = "midpoint"):
    if _ANCHOR_USES_TARGET:
        return AnchorSpec(target="line", index=index, mode=mode)
    return AnchorSpec(kind="line", index=index)


def area_anchor(index: int = 0, mode: str = "centroid"):
    if _ANCHOR_USES_TARGET:
        return AnchorSpec(target="area", index=index, mode=mode)
    return AnchorSpec(kind="area", index=index)


def point_anchor(index: int = 0):
    if _ANCHOR_USES_TARGET:
        return AnchorSpec(target="point", index=index, mode="exact")
    return AnchorSpec(kind="point", index=index)


def _dummy_energy(P, labels, scene, mask, cfg):
    return 0.0, np.zeros_like(P), {}


# ---------------------------------------------------------------------------
# Scene/label builders ------------------------------------------------------
# ---------------------------------------------------------------------------


def make_scene_and_label_with_dirty_polyline():
    """Create a scene with a dirty line and an associated label."""

    poly = [(0.0, 0.0), (0.0, 0.0), (1e-10, 0.0), (1.0, 0.0)]
    scene = Scene(frame_size=(800.0, 600.0), points=[], lines=[poly], areas=[])
    if _ANCHOR_USES_TARGET:  # legacy contract expects geometry on the label
        from types import SimpleNamespace

        lbl = SimpleNamespace(id=0, kind="line", anchor=line_anchor(0), polyline=poly, meta={})
    else:
        lbl = Label(id=0, kind="line", anchor=line_anchor(0), meta={})
    return scene, lbl


def test_geom_preproc_cleans_polyline_and_emits_event():
    P = np.zeros((1, 2))
    scene, lbl = make_scene_and_label_with_dirty_polyline()
    labels = [lbl]
    mask = np.array([True])
    cfg = {}
    pm = PassManager(cfg, ["geom_preproc"])
    wrapped = pm.wrap_energy(_dummy_energy)

    scene_obj = scene.model_dump() if hasattr(scene, "model_dump") else scene
    wrapped(P, labels, scene_obj, mask, cfg)

    gp = next(p for p in pm.passes if isinstance(p, GeomPreprocPass))
    # cleaned geometry may be stored in the scene or in the pass cache depending
    # on contract version.
    cleaned = None
    if scene.lines:
        cleaned = scene.lines[0]
    if cleaned != [(0.0, 0.0), (1.0, 0.0)] and 0 in gp.stage_cache:
        cached = gp.stage_cache[0][0]
        if isinstance(cached, dict):
            cleaned = cached.get("polyline", cached)
        else:
            cleaned = getattr(cached, "polyline", cached)
    assert cleaned == [(0.0, 0.0), (1.0, 0.0)]

    events = [e for e in pm.pop_events() if e.get("info") == "cleanup"]
    assert len(events) == 1
    m = events[0]["metrics"]
    assert m["collapsed"] == 1 and m["dropped_segments"] == 1 and m["affected_labels"] == 1


def test_energy_after_cleanup_no_shape_errors():
    P = np.zeros((1, 2))
    scene = Scene(frame_size=(800.0, 600.0), points=[], lines=[[(0.0, 0.0), (0.0, 0.0)]], areas=[])
    if _ANCHOR_USES_TARGET:
        from types import SimpleNamespace

        lbl = SimpleNamespace(
            id=0,
            kind="line",
            anchor=line_anchor(0),
            polyline=[(0.0, 0.0), (0.0, 0.0)],
            meta={},
        )
    else:
        lbl = Label(id=0, kind="line", anchor=line_anchor(0), meta={})

    mask = np.array([True])
    cfg = {}
    pm = PassManager(cfg, ["geom_preproc"])
    wrapped = pm.wrap_energy(energy_and_grad_full)
    scene_obj = scene.model_dump() if hasattr(scene, "model_dump") else scene
    E, G, comps = wrapped(P, [lbl], scene_obj, mask, cfg)
    assert G.shape == (1, 2)
    events = [e for e in pm.pop_events() if e.get("info") == "cleanup"]
    assert events[0]["metrics"]["affected_labels"] == 1
