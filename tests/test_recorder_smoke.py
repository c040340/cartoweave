from cartoweave.compute.run import solve

# ``AnchorSpec`` was introduced in the dataclass based SolvePack v2.  Older
# releases used ``Anchor`` with ``target/mode`` fields.  Import with graceful
# fallback so the tests remain compatible across versions.
try:  # pragma: no cover
    from cartoweave.contracts.solvepack import Scene, Label, AnchorSpec, SolvePack
    _ANCHOR_USES_TARGET = False
except Exception:  # pragma: no cover - legacy Pydantic contract
    from cartoweave.contracts.solvepack import Scene, Label, Anchor as AnchorSpec, SolvePack
    _ANCHOR_USES_TARGET = True


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


def _pack_with_two_stages():
    scene = Scene(frame_size=(10.0, 10.0), points=[(0.0, 0.0)], lines=[], areas=[])
    lbl = Label(id=0, kind="point", anchor=point_anchor(0), meta={})
    pack = SolvePack(
        L=1,
        P0=[(0.0, 0.0)],
        labels0=[lbl],
        active0=[True],
        scene0=scene,
        cfg={"compute": {}},
    )
    pack.__dict__["stages"] = [{}, {}]
    return pack


def test_recorder_produces_frames_and_events():
    pack = _pack_with_two_stages()
    vp = solve(pack)
    assert len(vp.frames) >= 2
    assert len(vp.events) >= 2
    summary = vp.summary
    assert summary["frames_captured"] == len(vp.frames)
    assert "terms_used" in summary
    assert "time_ms" in summary and summary["time_ms"] >= 0
    assert summary["global_iters"] >= len(vp.frames)

