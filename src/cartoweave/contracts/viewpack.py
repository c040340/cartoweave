# TODO(contracts): add strict validation to ViewPack/Frame similar to SolvePack
# - shape checks for arrays
# - non-negatives for sizes
# - required keys

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

Array2 = np.ndarray  # shape == (L, 2)


@dataclass
class Frame:
    """Single evaluation snapshot."""

    i: int  # evaluation index
    stage: int  # stage index
    E: float  # energy value
    P: Array2  # positions (L,2)
    G: Array2  # gradient (L,2)
    comps: Dict[str, Array2]  # per-term forces
    mask: np.ndarray  # active mask
    metrics: Dict[str, float] = field(default_factory=dict)  # extra metrics
    meta: Dict[str, Any] = field(default_factory=dict)  # passthrough metadata


@dataclass
class ViewPack:
    """Container returned by :func:`cartoweave.compute.run.solve`."""

    L: int  # number of labels
    mode: str  # solver mode used
    params_used: Dict[str, Any]  # final solver parameters
    terms_used: List[str]  # force term keys encountered
    frames: List[Frame]  # captured frames
    last: Frame  # reference to last frame
    summary: Dict[str, Any]  # aggregate statistics
