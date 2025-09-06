# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from .types import ViewPack


def to_old_payload(vp: ViewPack) -> Dict[str, Any]:
    """Convert a :class:`ViewPack` to the old visualization payload format.

    The returned structure contains minimal fields expected by earlier
    visualization utilities.
    """
    P_seq: List[np.ndarray] = []
    E_seq: List[float] = []
    G_seq: List[float] = []
    comps_seq: List[Dict[str, np.ndarray]] = []
    stage_seq: List[int] = []

    for f in vp.frames:
        P_seq.append(f.P)
        E_seq.append(f.E)
        G_seq.append(getattr(f, "Gnorm", 0.0))
        comps_seq.append(f.comps)
        stage_seq.append(getattr(f, "stage_id", 0))

    return {
        "P_seq": P_seq,
        "E_seq": E_seq,
        "G_seq": G_seq,
        "comps_seq": comps_seq,
        "stage_seq": stage_seq,
        "summary": dict(vp.summary),
    }
