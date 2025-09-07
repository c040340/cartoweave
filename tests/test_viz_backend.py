import os


def test_backend_fallback():
    from cartoweave.viz.backend import setup_matplotlib_backend

    os.environ.pop("MATPLOTLIB_BACKEND", None)
    bk = setup_matplotlib_backend(prefer="NonExistingBackend123", fallback="Agg")
    assert bk.lower() == "agg"

