"""Demonstrate the high-level :func:`solve_layout` API with mixed elements.

The scene contains one point, one polyline and one polygon. Each element has an
associated label anchored by element id. Seven actions include a mutate update
and activate/deactivate the labels in sequence. The result reports coordinates
at the end of every action.
"""

from __future__ import annotations

from cartoweave import solve_layout


def main() -> None:
    elements = {
        "points": [{"element_id": 0, "xy": [50.0, 50.0]}],
        "polylines": [{"element_id": 1, "polyline": [[10.0, 10.0], [60.0, 10.0]]}],
        "polygons": [{"element_id": 2, "polygon": [[20.0, 20.0], [40.0, 20.0], [30.0, 40.0]]}],
    }

    labels = [
        {"label_id": 0, "WH": [20.0, 10.0], "anchor": {"kind": "point", "element_id": 0}},
        {"label_id": 1, "WH": [30.0, 12.0], "anchor": {"kind": "line", "element_id": 1}},
        {"label_id": 2, "WH": [40.0, 15.0], "anchor": {"kind": "area", "element_id": 2}},
    ]

    actions = [
        {"t": 0, "op": "activate", "label_ids": [0]},
        {"t": 1, "op": "mutate", "label_ids": [0], "payload": {"WH": [25.0, 12.0]}},
        {"t": 2, "op": "activate", "label_ids": [1]},
        {"t": 3, "op": "activate", "label_ids": [2]},
        {"t": 4, "op": "deactivate", "label_ids": [0]},
        {"t": 5, "op": "deactivate", "label_ids": [1]},
        {"t": 6, "op": "deactivate", "label_ids": [2]},
    ]

    result = solve_layout(
        labels=labels,
        elements=elements,
        actions=actions,
        frame_size=(800.0, 600.0),
        #config_profile="../configs/compute",
    )

    for i, coords in enumerate(result.coords):
        print(f"action {i}: {coords}")


if __name__ == "__main__":  # pragma: no cover - example script
    main()
