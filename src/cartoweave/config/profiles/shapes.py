from __future__ import annotations
from typing import Dict, Any

# Keys that should be multiplied by sigma_scale when applying a profile
SIGMA_SCALE_KEYS = (
    "boundary.wall_eps",
    "eps.abs",
)

SHAPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        # no extends
        "fixed": {
            "ll.edge_power": 2.0,
            "boundary.wall_power": 3.0,
            "beta.softplus.dist": 2.0,
            "anchor.spring.alpha": 10.0,
            "area.cross.alpha": 10.0,
            "boundary.wall_eps": 0.5,
            "eps.abs": 0.5,
        },
    },
    "dense": {
        "extends": "default",
        "fixed": {
            "ll.edge_power": 3.0,
            "boundary.wall_power": 3.0,
            "beta.softplus.dist": 2.4,
            "anchor.spring.alpha": 12.0,
            "area.cross.alpha": 12.0,
            # eps values will be scaled by sigma_scale during application
        },
    },
    "sparse": {
        "extends": "default",
        "fixed": {
            "beta.softplus.dist": 1.6,
            "anchor.spring.alpha": 8.0,
            "area.cross.alpha": 8.0,
            # eps values will be scaled by sigma_scale during application
        },
    },
}
