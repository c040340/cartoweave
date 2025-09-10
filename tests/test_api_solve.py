import numpy as np
from cartoweave import solve

def test_solve_returns_coords_shape():
    labels = [
        {"id": 0, "xy0": [100.0, 200.0], "WH": [80.0, 24.0], "mode": "rect"},
        {"id": 1, "xy0": [220.0, 210.0], "WH": [60.0, 20.0], "mode": "rect"},
    ]
    elements = {"points": [], "polylines": [], "polygons": []}
    actions = [{"t": 0, "op": "activate", "target": "label", "ids": "all"}]

    res = solve(
        labels=labels,
        elements=elements,
        actions=actions,
        config_profile="configs/compute",
        deterministic_seed=123,
    )
    assert isinstance(res.coords, np.ndarray)
    assert res.coords.ndim == 3 and res.coords.shape[1:] == (2, 2)
