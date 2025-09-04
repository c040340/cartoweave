from copy import deepcopy
from typing import Mapping, Any


def deep_update(base: dict, override: Mapping[str, Any]) -> dict:
    """Return a copy of *base* recursively updated with *override*.

    Nested mappings in ``override`` are merged into copies of the corresponding
    sub-dictionaries in ``base``.  Non-mapping values replace the values in the
    copy.  The original ``base`` dictionary is left unchanged.
    """
    result = deepcopy(base)
    stack = [(result, override)]
    while stack:
        dst, src = stack.pop()
        for k, v in src.items():
            if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
                dst[k] = deepcopy(dst[k])
                stack.append((dst[k], v))
            else:
                dst[k] = deepcopy(v)
    return result
