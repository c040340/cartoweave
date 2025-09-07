"""End-to-end smoke test constructing a SolvePack and running compute."""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
from cartoweave.compute import solve
from cartoweave.config.loader import load_configs
from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.viz import interactive_view, merge_defaults, VIZ_DEFAULTS
from cartoweave.logging import init_logging_from_cfg
import matplotlib.pyplot as plt

headless = matplotlib.get_backend().lower() == "agg"
if headless:
    def _savefig(*args, **kwargs):
        plt.savefig("out_e2e.png", dpi=160, bbox_inches="tight")
    plt.show = _savefig


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

    # ---- 组装可视化所需数据 ----
    # 1) 轨迹：把每帧的 P 堆成 (T,N,2)
    traj = np.stack([f.P for f in view.frames], axis=0)  # (T, N, 2)

    # 2) 标签宽高：从初始 labels 取 WH，如果某些标签没设 WH，则填 1×1
    labels = pack.labels0  # list[dict]（pydantic 已序列化干净）
    rect_wh = []
    for lbl in labels:
        wh = lbl.WH
        if wh is None:
            rect_wh.append((1.0, 1.0))
        else:
            rect_wh.append(tuple(float(x) for x in wh))
    rect_wh = np.asarray(rect_wh, dtype=float)  # (N, 2)

    # 3) 场景底图与画布尺寸
    scene = pack.scene0
    points = scene.points
    lines = scene.lines
    areas = scene.areas
    W, H = scene.frame_size

    # 4) 可选：每帧分力与指标（显示在左/右面板）
    def force_getter(t: int):
        # 返回 dict[str, (N,2)]，interactive_view 会做筛选
        return view.frames[t].comps

    def metrics_getter(t: int):
        f = view.frames[t]
        return {"E": f.E, "Gnorm": f.Gnorm}

    # 5) 可选：把 viz.yaml 合并进默认可视化配置（如果你要调面板/分辨率）
    #   - 默认足够用，不传也可以
    viz_cfg = merge_defaults(VIZ_DEFAULTS)  # 或者从 configs/viz.yaml 读取后合并

    # ---- 启动交互式可视化 ----
    interactive_view(
        traj=traj,
        labels=labels,
        rect_wh=rect_wh,
        points=points,
        lines=lines,
        areas=areas,
        W=W,
        H=H,
        force_getter=force_getter,
        metrics_getter=metrics_getter,
        viz=viz_cfg,
    )


if __name__ == "__main__":
    main()
