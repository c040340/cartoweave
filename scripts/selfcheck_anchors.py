import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from cartoweave.compute.geometry import area_anchor_from_centroid_nearest_edge
from cartoweave.labels import anchor_xy


def main():
    fs = (800, 600)

    line = np.array([[0, 0], [100, 0], [100, 100]], float)
    qx, qy = anchor_xy("line", 0, {"lines": [line]}, fs)
    print("[line] anchor:", qx, qy)

    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    area_info = area_anchor_from_centroid_nearest_edge(square)
    print("[area] anchor:", area_info["qx"], area_info["qy"], "edge:", area_info["seg_index"], "n_in:", area_info["normal_in"])

    point = np.array([42, 17], float)
    px, py = anchor_xy("point", 0, {"points": np.array([point], float)}, fs)
    print("[point] anchor:", px, py)


if __name__ == "__main__":
    main()
