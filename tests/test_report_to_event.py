from cartoweave.compute.events import report_to_event
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.compute.solve import solve


def test_report_to_event_basic_dict():
    rep = {
        "it": 3,
        "g_norm": 0.1,
        "g_inf": 0.2,
        "step_size": 0.5,
        "ls_evals": 2,
        "delta_E": -0.4,
    }
    event, metrics = report_to_event(rep, "LBFGS")
    assert event["kind"] == "optimizer_step"
    assert event["algo"] == "LBFGS"
    assert event["iter_in_algo"] == 3
    assert event["step_size"] == 0.5
    assert event["ls_evals"] == 2
    assert event["delta_E"] == -0.4
    assert metrics["g_norm"] == 0.1
    assert metrics["g_inf"] == 0.2


def test_solve_emits_optimizer_step_event():
    pack = make_solvepack_from_data_defaults()
    vp = solve(pack)
    events = [ev for fr in vp.frames for ev in fr.meta["events"]]
    assert any(ev.get("kind") == "optimizer_step" for ev in events)
