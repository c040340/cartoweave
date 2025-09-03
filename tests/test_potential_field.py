import numpy as np
from cartoweave.engine.core_eval import scalar_potential_field
from cartoweave.config.presets import default_cfg
from cartoweave.config.utils import merge, viz


def test_scalar_potential_field_basic():
    scene = dict(
        frame=0,
        frame_size=(100, 80),
        labels_init=np.array([[30.0, 40.0]], float),
        WH=np.array([[10.0, 5.0]], float),
        anchors=np.array([[50.0, 40.0]], float),
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 4)),
        areas=np.zeros((0, 6)),
    )
    cfg = default_cfg()
    field = scalar_potential_field(scene, scene["labels_init"], cfg, resolution=3)
    expected_ny = max(1, int(round(3 * scene["frame_size"][1] / scene["frame_size"][0])))
    assert field.shape == (expected_ny, 3)
    assert np.isfinite(field).all()


def test_scalar_potential_field_cfg_resolution():
    scene = dict(
        frame=0,
        frame_size=(100, 80),
        labels_init=np.array([[30.0, 40.0]], float),
        WH=np.array([[10.0, 5.0]], float),
        anchors=np.array([[50.0, 40.0]], float),
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 4)),
        areas=np.zeros((0, 6)),
    )
    cfg = merge(default_cfg(), viz(field_resolution=7))
    field = scalar_potential_field(scene, scene["labels_init"], cfg)
    expected_ny = max(1, int(round(7 * scene["frame_size"][1] / scene["frame_size"][0])))
    assert field.shape == (expected_ny, 7)
    assert np.isfinite(field).all()
