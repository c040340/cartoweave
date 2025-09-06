import inspect, re
import cartoweave.compute.forces.area_cross as ac
import cartoweave.compute.forces.area_softout as aso


def test_area_cross_has_no_boolean_intersect_tokens():
    src = inspect.getsource(ac)
    forbidden = re.compile(r"(segment_intersects_rect|\bintersect\b|\boverlap\b)", re.I)
    assert forbidden.search(src) is None, "area_cross still contains boolean intersection logic"


def test_area_softout_uses_softclip_not_np_clip():
    src = inspect.getsource(aso)
    assert "np.clip" not in src and ".clip(" not in src, "area_softout should use softclip()"
