import numpy as np
import yaml
from pathlib import Path

from cartoweave.data.api import make_solvepack_from_data_defaults
from cartoweave.data.io import load_snapshot


def _minimal_generate_cfg(save_path: Path, seed: int = 123) -> dict:
    return {
        "data": {
            "source": "generate",
            "action_num": 1,
            "generate": {
                "counts": {"points": 2, "lines": 1, "areas": 1},
                "labels": 4,
                "frame_size": [64, 48],
                "seed": seed,
                "spacing": {"min_point_dist": 1.0, "margin": 0.0},
                "shapes": {"area_fallback": "polygon", "line_min_vertices": 2},
                "route_gen": {
                    "min_vertex_spacing_scale": 1.0,
                    "inset_margin_scale": 0.05,
                    "segment_len_scale": 0.10,
                    "angle_noise_sigma": 0.0,
                },
                "area_gen": {
                    "inset_margin_scale": 0.04,
                    "min_edge_spacing_scale": 0.04,
                    "n_vertices_min": 3,
                    "n_vertices_max": 3,
                },
                "anchors": {
                    "policy": "round_robin",
                    "modes": {"line": "midpoint", "area": "centroid"},
                },
                "save_path": str(save_path),
            },
        }
    }


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_generate_autosave_then_load_roundtrip(tmp_path):
    save_path = tmp_path / "autosave.json"
    cfg = _minimal_generate_cfg(save_path, seed=123)
    cfg_path = _write_cfg(tmp_path, cfg)

    pack_gen = make_solvepack_from_data_defaults(str(cfg_path))
    assert save_path.exists(), "Autosave file was not created"

    scene, P0, active0, labels, actions, action_num = load_snapshot(str(save_path))

    assert pack_gen.L == len(labels)
    assert np.asarray(pack_gen.P0).shape == P0.shape
    assert len(pack_gen.labels0) == len(labels)
    assert len(pack_gen.active0) == active0.shape[0]

    sg, sl = pack_gen.scene0, scene
    assert len(getattr(sg, "points", [])) == len(getattr(sl, "points", []))
    assert len(getattr(sg, "lines", [])) == len(getattr(sl, "lines", []))
    assert len(getattr(sg, "areas", [])) == len(getattr(sl, "areas", []))

    for i, (lg, ll) in enumerate(zip(pack_gen.labels0, labels)):
        assert lg.id == i and ll.id == i

    assert np.allclose(np.asarray(pack_gen.P0, dtype=float), P0)
    assert len(actions) == action_num == pack_gen.action_num


def test_autosave_overwrite(tmp_path):
    save_path = tmp_path / "autosave.json"

    cfg1 = _minimal_generate_cfg(save_path, seed=111)
    path1 = _write_cfg(tmp_path, cfg1)
    pack1 = make_solvepack_from_data_defaults(str(path1))
    assert save_path.exists(), "Autosave file was not created"

    cfg2 = _minimal_generate_cfg(save_path, seed=222)
    path2 = _write_cfg(tmp_path, cfg2)
    pack2 = make_solvepack_from_data_defaults(str(path2))

    _, P0_loaded, _, _, _, _ = load_snapshot(str(save_path))

    p0_loaded = np.asarray(P0_loaded, dtype=float)
    p0_first = np.asarray(pack1.P0, dtype=float)
    p0_second = np.asarray(pack2.P0, dtype=float)

    assert p0_loaded.shape == p0_second.shape == p0_first.shape
    assert not np.allclose(p0_first, p0_second)
    assert np.allclose(p0_loaded, p0_second)
