#!/usr/bin/env bash
set -euo pipefail
# Example: vary XY = 25,35,50 using base MAT from CONFIG (must exist)
python test/sensitivity_radiomap_res.py --scenario shanghai_single --power equal --sizes 25,35,50

