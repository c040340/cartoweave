from cartoweave.viz import show_vp, VizOpts
from cartoweave.compute.solve import solve
try:
    from tests.helpers import _make_pack  # type: ignore
except Exception:  # fallback if helpers module missing
    from tests.test_iter_capture import _make_pack

if __name__ == "__main__":
    # Use a small pack so that an extra final frame is emitted
    pack = _make_pack(5, every=3, final=True)
    vp = solve(pack)
    # Strip areas to avoid visualization issues in headless smoke test
    vp.sources.areas = []
    # Quick sanity check for required metadata
    assert all("pass_id" in fr.meta and "pass_name" in fr.meta for fr in vp.frames)
    assert any(fr.meta.get("frame_in_pass") == "final" for fr in vp.frames), "missing final frame"
    # Launch interactive viewer
    show_vp(vp, VizOpts())
