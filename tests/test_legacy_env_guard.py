import os
import pytest

def test_legacy_env_opt_in():
    assert os.getenv("LEGACY_FORCES_DIR") is None or isinstance(os.getenv("LEGACY_FORCES_DIR"), str)
