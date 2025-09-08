from dataclasses import dataclass

import numpy as np

from cartoweave.data.api import _derive_active0_from_actions


@dataclass
class Act:
    id: int
    t: float
    type: str


def test_active0_alignment():
    actions = [
        Act(id=0, t=-1.0, type="appear"),
        Act(id=1, t=1.0, type="appear"),
        Act(id=2, t=5.0, type="mutate"),
    ]
    active0 = _derive_active0_from_actions(3, actions)
    assert np.array_equal(active0, np.array([True, False, True]))
