from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.compute.solve import solve


def test_actions_only_solver_runs():
    sp = make_solvepack_from_data_defaults()
    assert len(sp.actions) > 0
    view = solve(sp)  # runs one step per action
    assert view is not None
