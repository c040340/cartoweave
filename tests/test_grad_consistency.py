"""
求解器烟雾测试：无 label 场景
---------------------------

目的：
- 验证在没有任何标签（labels）的极端场景下，求解器 API 不会报错，且返回形状正确。
- 这是一个防回退的“开机自检”用例，覆盖 solve_frame 的最小路径。
"""

import numpy as np
from cartoweave.api import solve_frame

def test_solve_no_labels():
    """
    当 labels 为空时：
    - 期望返回 P 的形状为 (0, 2)，info 字段可被忽略。
    - 任何 term 都不应被启用，也不应发生数值异常。
    """
    scene = dict(
        frame=0,
        frame_size=(1080, 1920),
        points=np.zeros((0, 2)),
        lines=np.zeros((0, 4)),
        areas=np.zeros((0, 6)),
        labels_init=np.zeros((0, 2)),
    )
    P, info = solve_frame(scene, cfg={}, mode="lbfgs")
    assert P.shape == (0, 2)
