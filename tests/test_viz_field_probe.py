import os
os.environ["MPLBACKEND"] = "Agg"  # pragma: no cover

import numpy as np
import matplotlib.pyplot as plt

from cartoweave.viz.panels import draw_field
from cartoweave.contracts.viewpack_v1 import ViewPack, VPFrame, VPPass, VPSources
from cartoweave.compute.forces import get_probe


def _make_viewpack(P, comps, compute_cfg):
    N = P.shape[0]
    frame = VPFrame(
        t=0,
        P=np.asarray(P, float),
        comps={k: np.asarray(v, float) for k, v in comps.items()},
        E=0.0,
        active_mask=np.ones(N, bool),
        meta={"global_iter": 0, "active_ids": list(range(N)), "active_count": N, "events": [{}]},
        metrics={},
        field=None,
    )
    vp = ViewPack(
        schema_version="viewpack-v1",
        N=N,
        labels=[{} for _ in range(N)],
        WH=None,
        frames=[frame],
        passes=[VPPass(0, "p", 0, 1)],
        sources=VPSources(points=np.zeros((0, 2)), lines=[], areas=[], frame_size=(100, 100)),
        defaults={"compute": compute_cfg},
        aux={},
    )
    return vp


def _field_from_ax(ax):
    assert ax.images, "field image should exist"
    return ax.images[0].get_array().copy()


def test_focus_field_invariant():
    cfg = {"public": {"forces": {"focus": {"attract": {"enable": True}}}}}
    vp1 = _make_viewpack(np.array([[10.0, 20.0]]), {"focus.attract": np.zeros((1, 2))}, cfg)
    fig, ax = plt.subplots()
    draw_field(ax, vp1, 0, {"field": {"terms": ["focus.attract"], "resolution": 16}})
    f1 = _field_from_ax(ax)
    plt.close(fig)

    vp2 = _make_viewpack(np.array([[80.0, 60.0]]), {"focus.attract": np.zeros((1, 2))}, cfg)
    fig, ax = plt.subplots()
    draw_field(ax, vp2, 0, {"field": {"terms": ["focus.attract"], "resolution": 16}})
    f2 = _field_from_ax(ax)
    plt.close(fig)

    assert np.allclose(f1, f2)


def test_ll_rect_changes_and_focus_stable():
    cfg = {
        "public": {
            "forces": {
                "focus": {"attract": {"enable": True}},
                "ll": {"rect": {"enable": True}},
            }
        }
    }
    P1 = np.array([[10.0, 10.0], [30.0, 10.0]])
    comps = {"focus.attract": np.zeros((2, 2)), "ll.rect": np.zeros((2, 2))}
    vp1 = _make_viewpack(P1, comps, cfg)
    fig, ax = plt.subplots()
    draw_field(ax, vp1, 0, {"field": {"terms": ["ll.rect"], "resolution": 16}})
    ll1 = _field_from_ax(ax)
    plt.close(fig)

    P2 = np.array([[20.0, 10.0], [30.0, 10.0]])
    vp2 = _make_viewpack(P2, comps, cfg)
    fig, ax = plt.subplots()
    draw_field(ax, vp2, 0, {"field": {"terms": ["ll.rect"], "resolution": 16}})
    ll2 = _field_from_ax(ax)
    plt.close(fig)
    assert not np.allclose(ll1, ll2)

    fig, ax = plt.subplots()
    draw_field(ax, vp1, 0, {"field": {"terms": ["focus.attract"], "resolution": 16}})
    ff1 = _field_from_ax(ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    draw_field(ax, vp2, 0, {"field": {"terms": ["focus.attract"], "resolution": 16}})
    ff2 = _field_from_ax(ax)
    plt.close(fig)
    assert np.allclose(ff1, ff2)


def test_field_fallback_title():
    cfg = {"public": {"forces": {}}}
    vp = _make_viewpack(np.array([[0.0, 0.0]]), {"unknown.term": np.zeros((1, 2))}, cfg)
    fig, ax = plt.subplots()
    draw_field(
        ax,
        vp,
        0,
        {"field": {"terms": ["unknown.term"], "fallback_splat": True, "resolution": 16}},
    )
    title = ax.get_title().lower()
    plt.close(fig)
    assert "fallback" in title


def test_basic_probes_nonzero():
    """Ensure representative probes yield non-zero fields."""

    pl = get_probe("pl.rect")
    F = pl({"points": np.array([[0.0, 0.0]])}, {}, np.array([[1.0, 0.0]]))
    assert np.linalg.norm(F) > 0.0

    ln = get_probe("ln.rect")
    F = ln({"lines": [np.array([[0.0, 0.0], [0.0, 1.0]])]}, {}, np.array([[1.0, 0.5]]))
    assert np.linalg.norm(F) > 0.0

    disk = get_probe("ll.disk")
    scene = {"labels_xy": np.array([[0.0, 0.0]]), "WH": np.array([[1.0, 1.0]])}
    F = disk(scene, {}, np.array([[2.0, 0.0]]))
    assert np.linalg.norm(F) > 0.0
