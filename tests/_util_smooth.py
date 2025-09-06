# pragma: no cover
import numpy as np
from typing import Callable, Dict, Any, List

from cartoweave.compute.eval import energy_and_grad_full


# --- scene & initial positions -------------------------------------------------

def build_scene_payload(L: int, frame_size: tuple[int, int], rng: np.random.Generator) -> Dict[str, Any]:
    """Construct a minimal scene payload accepted by :class:`SolvePack`.

    Parameters
    ----------
    L : int
        Number of labels.
    frame_size : tuple[int, int]
        ``(W, H)`` in pixels.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """
    labels = [{"anchor_kind": "none", "anchor_index": -1} for _ in range(L)]
    widths = rng.uniform(80.0, 160.0, size=(L, 1))
    heights = np.full((L, 1), 24.0)
    WH = np.concatenate([widths, heights], axis=1)
    scene = {
        "frame_size": tuple(int(v) for v in frame_size),
        "labels": labels,
        "WH": WH,
        "points": np.zeros((0, 2), float),
        "lines": [],
        "areas": [],
    }
    return scene


def build_P0(L: int, frame_size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Random initial positions possibly outside the frame."""
    W, H = frame_size
    x = rng.uniform(-0.5 * W, 1.5 * W, size=(L, 1))
    y = rng.uniform(-0.5 * H, 1.5 * H, size=(L, 1))
    return np.concatenate([x, y], axis=1)


# --- stage helpers -------------------------------------------------------------

def single_stage(iters: int) -> List[Dict[str, Any]]:
    return [{"iters": int(iters)}]


def three_stage_masks(L: int) -> List[Dict[str, Any]]:
    n1 = max(1, L // 3)
    n2 = max(n1, (2 * L) // 3)
    m1 = np.zeros(L, bool)
    m2 = np.zeros(L, bool)
    m3 = np.ones(L, bool)
    m1[:n1] = True
    m2[:n2] = True
    stages = [
        {"iters": 10, "mask_override": m1},
        {"iters": 10, "mask_override": m2},
        {"iters": 10, "mask_override": m3},
    ]
    return stages


# --- energy/gradient -----------------------------------------------------------

def get_energy_grad_callable() -> Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], tuple[float, np.ndarray]]:
    """Return the canonical energy+grad entry used by the solver.

    The solver uses :func:`energy_and_grad_full` from ``compute.eval`` with an
    explicit ``active_mask``. This helper wraps it to match a simplified
    ``(scene, P, cfg) -> (E, G)`` signature for tests.
    """

    def _call(scene: Dict[str, Any], P: np.ndarray, cfg: Dict[str, Any]):
        mask = np.ones(P.shape[0], bool)
        E, G, _, _ = energy_and_grad_full(P, scene, mask, cfg or {})
        return float(E), G

    return _call


# --- numerical gradient -------------------------------------------------------

def numgrad_central(
    E_fn: Callable[[Dict[str, Any], np.ndarray, Dict[str, Any]], tuple[float, np.ndarray]],
    scene: Dict[str, Any],
    P: np.ndarray,
    cfg: Dict[str, Any],
    h: float = 1e-3,
    max_dims: int = 48,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Central-difference numerical gradient on random subset of dims."""
    rng = rng or np.random.default_rng()
    flat = P.reshape(-1).copy()
    total = flat.size
    idxs = np.arange(total)
    rng.shuffle(idxs)
    idxs = idxs[: min(max_dims, total)]
    g = np.full(total, np.nan, float)
    for k in idxs:
        p_plus = flat.copy()
        p_minus = flat.copy()
        p_plus[k] += h
        p_minus[k] -= h
        E_plus, _ = E_fn(scene, p_plus.reshape(P.shape), cfg)
        E_minus, _ = E_fn(scene, p_minus.reshape(P.shape), cfg)
        g[k] = (E_plus - E_minus) / (2.0 * h)
    return g.reshape(P.shape)


# --- misc utilities -----------------------------------------------------------

def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)


def sign_change_count(series: np.ndarray) -> int:
    s = np.sign(series)
    prev = 0.0
    cnt = 0
    for val in s:
        if val == 0:
            continue
        if prev != 0 and val != prev:
            cnt += 1
        prev = val
    return cnt

