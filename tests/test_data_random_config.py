import numpy as np
from cartoweave.config.loader import load_configs
from cartoweave.data.random import generate_scene


def _avg_line_length(scene):
    lens = []
    for ln in scene["lines"]:
        seg = np.diff(np.asarray(ln, float), axis=0)
        lens.append(float(np.linalg.norm(seg, axis=1).sum()))
    return float(np.mean(lens)) if lens else 0.0


def test_route_gen_length_scale(tmp_path):
    bundle = load_configs()
    cfg = bundle.model_dump(exclude_unset=False, exclude_defaults=False)
    assert cfg["data"]["random"]["frame"]["width"] == 1920
    assert cfg["data"]["random"]["frame"]["height"] == 1080
    scene0 = generate_scene(gen_cfg=cfg["data"]["random"], seed=0)
    L0 = _avg_line_length(scene0)

    override = tmp_path / "solver.tuning.yaml"
    override.write_text(
        """tuning:\n  data:\n    random:\n      route_gen:\n        mean_length_scale: 0.40\n"""
    )
    bundle2 = load_configs(tuning_path=str(override))
    cfg2 = bundle2.model_dump(exclude_unset=False, exclude_defaults=False)
    scene1 = generate_scene(gen_cfg=cfg2["data"]["random"], seed=0)
    L1 = _avg_line_length(scene1)
    assert L1 > L0
