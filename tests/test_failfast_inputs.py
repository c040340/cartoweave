import numpy as np
import pytest
from cartoweave.contracts.solvepack import SolvePack


def test_validate_shapes(scene):
    with pytest.raises(ValueError):
        SolvePack(L=3, P0=np.zeros((2, 2)), active_mask0=np.ones((3,), bool), scene=scene)
    with pytest.raises(ValueError):
        SolvePack(L=3, P0=np.zeros((3, 2)), active_mask0=np.ones((2,), bool), scene=scene)
