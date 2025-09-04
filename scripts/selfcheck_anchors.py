import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from cartoweave.layout_utils.geometry import (
    poly_centroid,
    project_point_to_polyline,
    area_anchor_from_centroid_nearest_edge,
)
from cartoweave.labels import anchor_xy


def main():
    FS = (800, 600)

    L = np.array([[0, 0], [100, 0], [100, 100]], float)
    qx, qy = anchor_xy("line", 0, {"lines": [L]}, FS)
    print("[line] anchor:", qx, qy)

    S = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    A = area_anchor_from_centroid_nearest_edge(S)
    print("[area] anchor:", A["qx"], A["qy"], "edge:", A["seg_index"], "n_in:", A["normal_in"])

    P = np.array([42, 17], float)
    px, py = anchor_xy("point", 0, {"points": np.array([P], float)}, FS)
    print("[point] anchor:", px, py)


if __name__ == "__main__":
    main()
