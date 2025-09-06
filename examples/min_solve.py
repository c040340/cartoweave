import json

from cartoweave.data.api import load_solvepack_from_file
from cartoweave.compute.run import solve
from cartoweave.config.bridge import translate_legacy_keys

cfg = json.load(open("examples/configs/compute_min.json", "r", encoding="utf-8"))
cfg = translate_legacy_keys(cfg)

sp = load_solvepack_from_file(
    "examples/scenes/scene_min.json",
    solver_cfg=cfg,
    seed=0,
)
view = solve(sp)
print("frames:", view.summary.get("frames_captured"), "E_last:", view.summary.get("E_last"))
