import numpy as np
import pytest
from cartoweave.compute import SolvePack

def test_validate_shapes(scene):
    with pytest.raises(AssertionError):
        SolvePack(L=3, P0=np.zeros((2,2)), active_mask0=np.ones((3,),bool), scene=scene).validate()
    with pytest.raises(AssertionError):
        SolvePack(L=3, P0=np.zeros((3,2)), active_mask0=np.ones((2,),bool), scene=scene).validate()
