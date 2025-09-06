# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from cartoweave.contracts.viewpack import ViewPack


def to_old_payload(vp: ViewPack) -> Dict[str, Any]:
    """Convert a :class:`ViewPack` to the old visualization payload format.

    The returned structure contains minimal fields expected by earlier
    visualization utilities.
    """
    P_seq: List[np.ndarray] = []
    E_seq: List[float] = []
    G_seq: List[np.ndarray] = []
    comps_seq: List[Dict[str, np.ndarray]] = []
    mask_seq: List[np.ndarray] = []
    stage_seq: List[int] = []

    for f in vp.frames:
        P_seq.append(f.P)
        E_seq.append(f.E)
        G_seq.append(f.G)
        comps_seq.append(f.comps)
        mask_seq.append(f.mask)
        stage_seq.append(f.stage)

    return {
        "P_seq": P_seq,
        "E_seq": E_seq,
        "G_seq": G_seq,
        "comps_seq": comps_seq,
        "mask_seq": mask_seq,
        "stage_seq": stage_seq,
        "summary": dict(vp.summary),
        "L": vp.L,
        "mode": vp.mode,
        "terms": vp.terms_used,
    }
