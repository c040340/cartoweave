#!/usr/bin/env bash
set -euo pipefail
echo "[dev] pytest collect size"
pytest -q --collect-only | tee .pytest_collect.txt >/dev/null || true
echo "[dev] number of collected test items (rough):"
grep -c "::" .pytest_collect.txt || true
echo "[dev] grep risky tokens"
rg -n "stages|mask_override|active_mask0_from_stages" src || true
