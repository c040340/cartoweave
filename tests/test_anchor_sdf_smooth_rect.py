import numpy as np

from cartoweave.compute.geometry_sdf_smooth import (
    huber_prime,
    rect_implicit_smooth_world,
    sdf_from_implicit,
    smooth_abs,
)


def _inside(p, c, e, R=None):
    if R is not None:
        pl = (p - c) @ R.T
    else:
        pl = p - c
    return (np.abs(pl[..., 0]) <= e[0]) & (np.abs(pl[..., 1]) <= e[1])


def test_rect_sdf_axis_aligned_sign_and_norm():
    c = np.array([0.0, 0.0])
    e = np.array([1.0, 1.5])
    pts = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [0.5, 0.5],
        [0.9, 1.4],
        [1.2, 1.4],
    ])
    F, grad = rect_implicit_smooth_world(pts, c, e)
    s, n = sdf_from_implicit(F, grad)
    inside = _inside(pts, c, e)
    assert np.all((s < 0) == inside)
    norms = np.linalg.norm(n, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_rect_sdf_rotated_sign_and_norm():
    c = np.array([0.0, 0.0])
    e = np.array([1.0, 1.0])
    th = np.pi / 4.0
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [0.5, 0.5],
        [0.8, 0.0],
    ])
    F, grad = rect_implicit_smooth_world(pts, c, e, R=R)
    s, n = sdf_from_implicit(F, grad)
    inside = _inside(pts, c, e, R)
    assert np.all((s < 0) == inside)
    norms = np.linalg.norm(n, axis=1)
    mask = norms > 0
    assert np.allclose(norms[mask], 1.0, atol=1e-5)


def test_rect_sdf_continuity_across_corner():
    c = np.array([0.0, 0.0])
    e = np.array([1.0, 1.0])
    t = np.linspace(0.8, 1.2, 200)
    pts = np.stack([t, t], axis=1)
    F, grad = rect_implicit_smooth_world(pts, c, e)
    s, _ = sdf_from_implicit(F, grad)
    L = np.abs(s)
    assert np.all(np.isfinite(L))
    diffs = np.diff(L)
    assert np.max(np.abs(diffs)) < 0.2


def test_huber_reduces_force_near_r0():
    c = np.array([0.0, 0.0])
    e = np.array([1.0, 1.0])
    A = np.array([[2.5, 0.0]])
    F, grad = rect_implicit_smooth_world(A, c, e)
    s, n = sdf_from_implicit(F[0], grad[0])
    L = smooth_abs(s, 1e-6)
    r0 = 0.1
    d = L - r0
    sgn = s / (L + 1e-12)

    def mag(delta: float) -> float:
        d_eff = huber_prime(d, delta) if delta > 0 else d
        return np.linalg.norm(-1.0 * d_eff * sgn * n)

    mag_on = mag(0.25)
    mag_off = mag(0.0)
    assert mag_on < mag_off
