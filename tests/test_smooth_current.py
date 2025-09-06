# pragma: no cover
import numpy as np
import pytest
import os
import sys

from cartoweave.contracts.solvepack import SolvePack

sys.path.append(os.path.dirname(__file__))
from _util_smooth import (
    build_scene_payload,
    build_P0,
    single_stage,
    three_stage_masks,
    get_energy_grad_callable,
    numgrad_central,
    rel_err,
    sign_change_count,
)


FRAME = (1280, 800)
SEEDS = [7, 19]
SCALES = [4, 16, 48]
H_STEP = 1e-3
MAX_DIMS = 36
GRAD_THRESH = dict(median=1e-4, p95=5e-4, max=1e-3)
LINE_SWEEP_N = 60
LINE_SWEEP_RANGE = (-3.0, +3.0)
LINE_JUMPS_MAX = 6


# ------------------------------------------------------------------------------
# A) Grad-check on conservative total (analytic vs numerical)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("L", SCALES)
@pytest.mark.parametrize("seed", SEEDS)
def test_grad_check_conservative_total(L, seed):
    rng = np.random.default_rng(seed)
    scene = build_scene_payload(L, FRAME, rng)
    P0 = build_P0(L, FRAME, rng)
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=np.ones(L, bool),
        scene=scene,
        cfg={},
        stages=single_stage(1),
        passes=["schedule"],
    )
    Efn = get_energy_grad_callable()
    E, G = Efn(sp.scene, sp.P0, sp.cfg)
    G_num = numgrad_central(Efn, sp.scene, sp.P0, sp.cfg, h=H_STEP, max_dims=MAX_DIMS, rng=rng)
    r = rel_err(G, G_num)
    assert np.nanmedian(r) <= GRAD_THRESH["median"]
    assert np.nanpercentile(r, 95) <= GRAD_THRESH["p95"]
    assert np.nanmax(r) <= GRAD_THRESH["max"]


# ------------------------------------------------------------------------------
# B) Line-probe smoothness (energy continuity / limited curvature jumps)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("L", [16])
@pytest.mark.parametrize("seed", SEEDS)
def test_energy_line_probe_smooth(L, seed):
    rng = np.random.default_rng(seed)
    scene = build_scene_payload(L, FRAME, rng)
    P0 = build_P0(L, FRAME, rng)
    sp = SolvePack(
        L=L,
        P0=P0,
        active_mask0=np.ones(L, bool),
        scene=scene,
        cfg={},
        stages=single_stage(1),
        passes=["schedule"],
    )
    Efn = get_energy_grad_callable()
    i = rng.integers(0, L)
    axis = rng.integers(0, 2)
    offsets = np.linspace(*LINE_SWEEP_RANGE, LINE_SWEEP_N)
    energies = []
    for off in offsets:
        P = sp.P0.copy()
        P[i, axis] = P[i, axis] + off
        E, _ = Efn(sp.scene, P, sp.cfg)
        assert np.isfinite(E), "Energy must remain finite along line sweep"
        energies.append(E)
    energies = np.asarray(energies)
    secdiff = np.diff(energies, n=2)
    assert np.isfinite(secdiff).all()
    assert sign_change_count(secdiff) <= LINE_JUMPS_MAX


# ------------------------------------------------------------------------------
# C) Extreme configurations are finite (no NaN/Inf)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_extremes_are_finite(seed):
    rng = np.random.default_rng(seed)
    L = 6
    scene = build_scene_payload(L, FRAME, rng)
    P0 = build_P0(L, FRAME, rng)

    P1 = P0.copy()
    if L >= 2:
        P1[1] = P1[0]

    P2 = P0.copy()
    W_half = scene["WH"][0, 0] * 0.5
    P2[0, 0] = -W_half + 1e-6

    scene_area = scene.copy()
    scene_area["areas"] = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)]

    Efn = get_energy_grad_callable()
    for P in (P0, P1, P2):
        E, G = Efn(scene, P, {})
        assert np.isfinite(E) and np.isfinite(G).all()
    E, G = Efn(scene_area, P0, {})
    assert np.isfinite(E) and np.isfinite(G).all()


# ------------------------------------------------------------------------------
# D) Single-stage vs multi-stage equivalence (same final active set)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("L", [24])
@pytest.mark.parametrize("seed", SEEDS)
def test_single_vs_multistage_equivalence(L, seed):
    rng = np.random.default_rng(seed)
    scene = build_scene_payload(L, FRAME, rng)
    P0 = build_P0(L, FRAME, rng)
    Efn = get_energy_grad_callable()

    def descend(P, stages, step=1e-2):
        P = P.copy()
        for st in stages:
            iters = st["iters"]
            mask = st.get("mask_override", np.ones(L, bool))
            for _ in range(iters):
                _, G = Efn(scene, P, {})
                Gm = G * mask[:, None]
                P = P - step * Gm
        E, _ = Efn(scene, P, {})
        return E

    E1 = descend(P0, single_stage(30))
    E2 = descend(P0, three_stage_masks(L))
    assert np.isfinite(E1) and np.isfinite(E2)
    assert abs(E1 - E2) <= 1e-2 * (1.0 + abs(E1))


# ------------------------------------------------------------------------------
# E) Determinism under fixed seeds
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("L", [16])
def test_determinism_fixed_seed(L):
    seed = 123
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    scene1 = build_scene_payload(L, FRAME, rng1)
    scene2 = build_scene_payload(L, FRAME, rng2)
    P01 = build_P0(L, FRAME, rng1)
    P02 = build_P0(L, FRAME, rng2)
    assert np.array_equal(scene1["WH"], scene2["WH"])
    assert np.array_equal(P01, P02)
    Efn = get_energy_grad_callable()
    E1, G1 = Efn(scene1, P01, {})
    E2, G2 = Efn(scene2, P02, {})
    assert np.isclose(E1, E2, rtol=1e-12, atol=1e-12)
    assert np.allclose(G1, G2, rtol=1e-12, atol=1e-12)

