import numpy as np


def as_nx2(x, N: int, name: str):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        if a.size == 2:
            a = a.reshape(1, 2)
        else:
            raise ValueError(f"{name}: expect 2 numbers, got shape {a.shape}")
    if a.shape == (1, 2):
        return np.repeat(a, N, axis=0)
    if a.shape == (N, 2):
        return a
    raise ValueError(f"{name}: shape {a.shape} incompatible with N={N}")
