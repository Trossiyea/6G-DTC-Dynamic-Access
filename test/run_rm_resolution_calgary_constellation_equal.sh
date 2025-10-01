#!/usr/bin/env bash
set -euo pipefail
python test/sensitivity_radiomap_res.py --scenario calgary_constellation --power equal --sizes 25,35,50

