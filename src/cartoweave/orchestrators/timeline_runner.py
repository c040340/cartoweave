import numpy as np
from cartoweave.orchestrators.timeline import run_timeline

scene = dict(
    frame=0, frame_size=(800, 600),
    labels_init=np.array([[200.,150.],[260.,150.]], float),
    WH=np.array([[60.,24.],[60.,24.]], float),
    points=np.zeros((0,2), float),   # 不用 points
    lines=np.zeros((0,4), float),
    areas=[{"polygon": np.array([[520,300],[340,404],[340,196]], float)}],
    labels=[{"anchor_kind":"none"},{"anchor_kind":"none"}],
    anchors=np.array([[180.,150.],[220.,150.]], float),
)

cfg = {
    # ll/pl/ln/boundary/area/focus：随你开关
    "ll.geom": "rect",
    "ll.k.repulse": 150.0, "ll.k.inside": 50.0,
    "boundary.k.wall": 80.0, "boundary.wall_eps": 0.3,
    "anchor.k.spring": 10.0,

    # Focus 椭圆 Huber（不依赖 points）
    "focus.k.attract": 30.0,
    "focus.center": np.array([400.,300.]),
    "focus.sigma.x": 80.0, "focus.sigma.y": 120.0,
    "focus.delta": 10.0,

    # 显式关掉 point↔label
    "pl.k.repulse": 0.0, "pl.k.inside": 0.0,
}

schedule = [
    {"name": "warmup_no_anchor", "scale": {"anchor.k.spring": 0.0}},
    # 可选：先弱化 ll 冲突，减小刚性；再恢复
    {"name": "ease_in_ll", "scale": {"ll.k.repulse": 0.5, "ll.k.inside": 0.5}},
    {"name": "main_solve"},  # 原始 cfg
]

P, info = run_timeline(scene, cfg, schedule)
print("P_final:", P)
for s in info["timeline"]:
    print(s["name"], "→", s["stage_info"])
