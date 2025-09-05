# tests/test_focus_huber.py
import numpy as np
from cartoweave.compute.eval import energy_and_grad_full

def test_focus_huber_anisotropy():
    scene = dict(
        frame=0, frame_size=(800, 600),
        labels_init=np.array([[300., 300.]], float),
        WH=np.array([[60., 24.]], float),
        labels=[{"anchor_kind":"none"}],
        points=np.zeros((0,2)), lines=np.zeros((0,2,2)),
        areas=np.zeros((0,6)), anchors=np.zeros((1,2)),
    )
    cfg = {
        "focus.k.attract": 50.0,
        "focus.center": np.array([300., 300.]),
        "focus.sigma.x": 50.0,   # x 方向更“紧”
        "focus.sigma.y": 150.0,  # y 方向更“宽”
        "focus.delta": 8.0,
    }

    # 在等距环上取两个点：沿 x 和沿 y
    P = scene["labels_init"].copy()
    P_x = P.copy(); P_x[0,0] += 60.0
    P_y = P.copy(); P_y[0,1] += 60.0

    mask = np.ones(P_x.shape[0], bool)
    _, Gx, _, _ = energy_and_grad_full(P_x, scene, mask, cfg)
    _, Gy, _, _ = energy_and_grad_full(P_y, scene, mask, cfg)

    # x 方向的恢复力（|Fx|）应大于 y 方向（因为 σx 更小 → 约束更强）
    Fx = abs(Gx[0,0])  # grad = -F
    Fy = abs(Gy[0,1])
    assert Fx > Fy, f"anisotropy violated: Fx={Fx}, Fy={Fy}"
