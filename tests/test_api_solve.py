import numpy as np
from cartoweave import solve_layout

def test_solve_returns_coords_shape():
    labels = [
        {
            "label_id": 0,
            "WH": [80.0, 24.0],
            "mode": "rect",
            "anchor": {"kind": "point", "element_id": 0},
        },
        {"label_id": 1, "xy0": [220.0, 210.0], "WH": [60.0, 20.0], "mode": "rect"},
    ]
    elements = {"points": [{"element_id": 0, "xy": [100.0, 200.0]}], "polylines": [], "polygons": []}
    actions = [
        {"t": 0, "op": "activate", "label_ids": "all"},
        {"t": 1, "op": "mutate", "label_ids": [1], "payload": {"WH": [60.0, 20.0]}},
    ]

    res = solve_layout(
        labels=labels,
        elements=elements,
        actions=actions,
        deterministic_seed=123,
    )
    assert isinstance(res.coords, np.ndarray)
    assert res.coords.ndim == 3 and res.coords.shape[1:] == (2, 2)
