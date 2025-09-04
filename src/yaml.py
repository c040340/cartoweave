"""A tiny YAML subset parser for tests.

Supports mappings, lists, scalars (numbers/strings/bools/null) and inline
``{a: b}`` / ``[1,2]`` forms. It is intentionally minimal and only handles
syntax used in the project configuration files."""
from __future__ import annotations

import re
from typing import Any, List, Tuple

__all__ = ["safe_load"]

_bool_map = {"true": True, "false": False}


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if token[0] in '"\'':
        return token[1:-1]
    if token in _bool_map:
        return _bool_map[token]
    if token in ("null", "None"):
        return None
    if token.startswith("[") and token.endswith("]"):
        inner = _split_items(token[1:-1])
        return [_parse_scalar(x) for x in inner]
    if token.startswith("{") and token.endswith("}"):
        items = _split_items(token[1:-1])
        out = {}
        for it in items:
            k, v = it.split(":", 1)
            out[k.strip()] = _parse_scalar(v)
        return out
    # numeric
    try:
        if "." in token or "e" in token or "E" in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _split_items(s: str) -> List[str]:
    items: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(s[start:i])
            start = i + 1
    items.append(s[start:])
    return items


def _parse_lines(lines: List[str], start: int = 0, indent: int = 0) -> Tuple[Any, int]:
    mapping = {}
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        cur_indent = len(line) - len(line.lstrip(" "))
        if cur_indent < indent:
            break
        line = line.strip()
        if line.startswith("- "):
            raise ValueError("Unexpected list item at root")
        if ":" not in line:
            i += 1
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            i += 1
        else:
            # determine block type
            i += 1
            if i < len(lines) and lines[i].lstrip().startswith("- "):
                items = []
                while i < len(lines):
                    l2 = lines[i]
                    if not l2.strip():
                        i += 1
                        continue
                    ind2 = len(l2) - len(l2.lstrip(" "))
                    if ind2 <= cur_indent:
                        break
                    l2 = l2.strip()
                    if not l2.startswith("- "):
                        break
                    item = l2[2:]
                    items.append(_parse_scalar(item))
                    i += 1
                mapping[key] = items
            else:
                submap, i = _parse_lines(lines, i, cur_indent + 2)
                mapping[key] = submap
    return mapping, i


def safe_load(stream: str | bytes) -> Any:
    if isinstance(stream, (bytes, bytearray)):
        text = stream.decode("utf-8")
    elif hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)
    # strip comments
    cleaned_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        cleaned_lines.append(line.rstrip())
    data, _ = _parse_lines(cleaned_lines)
    return data

