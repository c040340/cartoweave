import numpy as np
import pytest

from cartoweave.viz import metrics as vm


@pytest.fixture(autouse=True)
def reset_state():
    vm._prev_absF = None


class Cfg:
    viz_F_tol = 1e-3
    viz_topk = 3


def _labels():
    return [
        {"kind": "rect"},
        {"kind": "rect"},
        {"mode": "circle"},
        {"kind": "rect", "locked": True},
    ]


def test_metrics_pairs_and_styles_exclude_nonmovables():
    forces = np.array([[1e-2, 0.0], [5e-4, 2e-4], [0.0, 0.0], [10.0, 10.0]])
    solver_info = {"solver": "lbfgs", "g_inf": 9.8e-4, "gtol": 1e-3, "iter": 5, "iter_max": 60}
    m = vm.collect_solver_metrics(None, forces, {}, _labels(), solver_info, Cfg())
    assert m["F_inf"] == pytest.approx(1e-2)
    assert m["ui"]["F_ratio"] == pytest.approx(10.0)
    assert m["ui"]["g_ratio"] == pytest.approx(0.98)
    s1 = vm.style_by_ratio(0.98)
    s2 = vm.style_by_ratio(10.0)
    assert s1["bold"] is True and s1["color"] == "#1f9d55"
    assert s2["bold"] is False and s2["color"] == "#e53e3e"


def test_topk_forces_are_movable_only():
    forces = np.array([[1e-2, 0.0], [5e-4, 2e-4], [0.0, 0.0], [10.0, 10.0]])
    solver_info = {"solver": "lbfgs", "g_inf": 9.8e-4, "gtol": 1e-3}
    m = vm.collect_solver_metrics(None, forces, {}, _labels(), solver_info, Cfg())
    top = m["top_force_labels"]
    indices = [i for i, _ in top]
    assert indices == [0, 1]
    assert top[0][1] >= top[1][1]


def test_deltaF_and_ratio():
    forces1 = np.array([[1e-2, 0.0], [5e-4, 2e-4], [0.0, 0.0], [10.0, 10.0]])
    solver_info = {"solver": "lbfgs", "g_inf": 9.8e-4, "gtol": 1e-3}
    m1 = vm.collect_solver_metrics(None, forces1, {}, _labels(), solver_info, Cfg())
    assert np.allclose(m1["deltaF"], 0.0)
    assert np.allclose(m1["deltaF_over_F"], 0.0)

    forces2 = forces1.copy()
    forces2[0, 0] = 1.2e-2
    m2 = vm.collect_solver_metrics(None, forces2, {}, _labels(), solver_info, Cfg())
    assert m2["deltaF"][0] == pytest.approx(2e-3)
    assert m2["deltaF_over_F"][0] == pytest.approx(0.2, rel=1e-6)
    assert m2["F_inf"] == pytest.approx(1.2e-2)


def test_solver_info_passthrough():
    forces = np.zeros((1, 2))
    labels = [{"kind": "rect"}]
    cfg = Cfg()
    s1 = {"solver": "semi", "g_inf": 2e-4, "gtol": 1e-3, "iter": 1, "iter_max": 10}
    out1 = vm.collect_solver_metrics(None, forces, {}, labels, s1, cfg)
    assert out1["gtol"] == pytest.approx(1e-3)
    s2 = {"solver": "hybrid", "g_inf": 5e-4, "gtol": 1e-3, "stage": "lbfgs→semi", "iter": 2, "iter_max": 10}
    out2 = vm.collect_solver_metrics(None, forces, {}, labels, s2, cfg)
    assert out2["gtol"] == pytest.approx(1e-3)


def test_edgecases_zero_and_small_tol():
    class TinyCfg:
        viz_F_tol = 0.0
        viz_topk = 3
    forces = np.zeros((2, 2))
    labels = [{"kind": "rect"}, {"kind": "rect"}]
    solver_info = {"solver": "lbfgs", "g_inf": 0.0, "gtol": 0.0}
    out = vm.collect_solver_metrics(None, forces, {}, labels, solver_info, TinyCfg())
    assert out["F_inf"] == 0.0
    assert out["ui"]["F_ratio"] == 0.0
    assert out["ui"]["g_ratio"] == 0.0
    assert out["ui"]["F_style"]["bold"] is True
    assert out["ui"]["F_style"]["color"] == "#1f9d55"
    assert out["ui"]["g_style"]["bold"] is True
    assert out["ui"]["g_style"]["color"] == "#1f9d55"
    assert not np.isnan(out["ui"]["F_ratio"]) and not np.isnan(out["ui"]["g_ratio"])
