"""End-to-end demo using the new data API with visualisation."""

from cartoweave.data.api import build_solvepack_from_config
from cartoweave.compute.run import solve
from cartoweave.viz.build_viz_payload import build_viz_payload
from cartoweave.viz.view import interactive_view
from cartoweave.viz.defaults import VIZ_DEFAULTS
from cartoweave.viz.backend import use_compatible_backend


def main() -> None:
    sp = build_solvepack_from_config(
        "../examples/configs/demo.yaml",
        overrides={"data": {"regen": True}},
        seed=123,
    )
    view = solve(sp)
    use_compatible_backend()
    payload = build_viz_payload(view, sp.scene, sp.stages)
    interactive_view(payload, VIZ_DEFAULTS)


if __name__ == "__main__":
    main()
