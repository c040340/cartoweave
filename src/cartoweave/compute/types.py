# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TypedDict  # noqa: F401
import numpy as np

Array2 = np.ndarray  # shape == (L, 2)

# energy 回调签名：返回 (E, G_full, comps_full, meta)
EnergyFn = Callable[
    [Array2, object, np.ndarray, dict],
    Tuple[float, Array2, Dict[str, Array2], dict],
]

@dataclass
class Frame:
    i: int
    stage: int
    E: float
    P: Array2
    G: Array2
    comps: Dict[str, Array2]
    mask: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

@dataclass
class ViewPack:
    L: int
    mode: str
    params_used: Dict
    terms_used: List[str]
    frames: List[Frame]
    last: Frame
    summary: Dict
