#!/usr/bin/env bash
set -euo pipefail

echo "Running Shanghai Single (equal)";          bash $(dirname "$0")/run_shanghai_single_equal.sh
echo "Running Shanghai Single (waterfill)";      bash $(dirname "$0")/run_shanghai_single_waterfill.sh
echo "Running Shanghai Constellation (equal)";   bash $(dirname "$0")/run_shanghai_constellation_equal.sh
echo "Running Shanghai Constellation (waterfill)"; bash $(dirname "$0")/run_shanghai_constellation_waterfill.sh
echo "Running Calgary Single (equal)";           bash $(dirname "$0")/run_calgary_single_equal.sh
echo "Running Calgary Single (waterfill)";       bash $(dirname "$0")/run_calgary_single_waterfill.sh
echo "Running Calgary Constellation (equal)";    bash $(dirname "$0")/run_calgary_constellation_equal.sh
echo "Running Calgary Constellation (waterfill)"; bash $(dirname "$0")/run_calgary_constellation_waterfill.sh

echo "All scenarios completed. See test/out/ for results."

