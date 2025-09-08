"""End-to-end smoke test constructing a SolvePack and running compute."""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
from cartoweave.compute import solve
from cartoweave.config.loader import load_configs
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.logging import init_logging_from_cfg
from cartoweave.viz import show_vp, render_frame, VizOpts
from cartoweave.logging import init_logging_from_cfg

headless = matplotlib.get_backend().lower() == "agg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    overrides = json.loads(args.overrides) if args.overrides else None
    compute_cfg = load_configs(overrides=overrides)["compute"]
    pack = make_solvepack_from_data_defaults(
        compute_cfg=compute_cfg, data_path="configs/data.yaml"
    )
    pack.validate()
    init_logging_from_cfg(getattr(pack, "cfg", None))
    view = solve(pack)
    last = view.frames[-1]
    print(f"[e2e] N={pack.L} frames={len(view.frames)} final_E={last.E:.3g}")
    print(f"[e2e] P_norm={float(np.linalg.norm(last.P)):.3g}")

    if headless:
        # 导出静态帧供 CI 等环境使用
        fig = render_frame(view, t=view.num_frames() - 1)
        fig.savefig("last.png", dpi=180, bbox_inches="tight")
    else:
        # 直接使用新的 ViewPack 浏览器
        show_vp(view, VizOpts())


if __name__ == "__main__":
    main()
