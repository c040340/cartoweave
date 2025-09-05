import json, numpy as np
from cartoweave.compute.pack import SolvePack
from cartoweave.compute.run import solve
from cartoweave.config.bridge import translate_legacy_keys

cfg = json.load(open("examples/configs/compute_min.json", "r", encoding="utf-8"))
cfg = translate_legacy_keys(cfg)
scene = json.load(open("examples/scenes/scene_min.json", "r", encoding="utf-8"))

L = len(scene["labels_init"])
sp = SolvePack(
    L=L,
    P0=np.asarray(scene["labels_init"], float),
    active_mask0=np.ones(L, bool),
    scene=scene,
    cfg=cfg,
    stages=[{"iters": 5, "solver": "lbfgs"}],
    passes=["schedule", "capture"],
)
view = solve(sp)
print("frames:", view.summary.get("frames_captured"), "E_last:", view.summary.get("E_last"))
