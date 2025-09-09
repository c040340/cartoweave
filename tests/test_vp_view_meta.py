from types import SimpleNamespace

from cartoweave.viz.vp_view import format_meta


def test_format_meta_uses_first_optimizer_event_and_label_text():
    labels = [SimpleNamespace(id=0, kind="point"), SimpleNamespace(id=1, kind="area")]
    vp = SimpleNamespace(labels=labels)
    meta = {
        "events": [
            {"kind": "optimizer_step", "algo": "semi_newton", "iter_in_algo": 0},
            {"pass": "action", "info": "mutate", "label_id": 1},
            {"kind": "optimizer_step", "algo": "lbfgs", "iter_in_algo": 1},
        ]
    }
    txt = format_meta(meta, vp)
    lines = txt.splitlines()
    assert lines[0] == "semi_newton:0"
    assert lines[1] == "mutate a1"
