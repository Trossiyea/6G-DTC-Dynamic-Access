#!/bin/bash
# =============================================================================
# NR-NTN Downlink Simulator - Comprehensive Test Suite
# =============================================================================
#
# Usage:
#   ./run_all_tests.sh           # Run all tests (unit + scenarios)
#   ./run_all_tests.sh --quick   # Quick mode: unit tests + 2 scenarios
#   ./run_all_tests.sh --unit    # Unit tests only
#   ./run_all_tests.sh --scenarios # Simulation scenarios only
#
# Requirements:
#   - Conda environment 'DtC' with all dependencies
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Parse arguments
MODE="all"
if [[ "$1" == "--quick" ]]; then
    MODE="quick"
elif [[ "$1" == "--unit" ]]; then
    MODE="unit"
elif [[ "$1" == "--scenarios" ]]; then
    MODE="scenarios"
fi

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_subheader() {
    echo ""
    echo -e "${YELLOW}----------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}----------------------------------------------------------------------${NC}"
}

run_test() {
    local name="$1"
    local cmd="$2"

    echo -n "  Testing $name... "
    if eval "$cmd" > /tmp/test_output.txt 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "    Error output:"
        tail -10 /tmp/test_output.txt | sed 's/^/    /'
        ((FAILED++))
        return 1
    fi
}

# =============================================================================
# Environment Setup
# =============================================================================

print_header "NR-NTN Downlink Simulator - Test Suite"
echo "Mode: $MODE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

# Activate conda environment
print_subheader "Activating Conda Environment (DtC)"

# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
else
    echo -e "${RED}Error: Cannot find conda.sh${NC}"
    exit 1
fi

conda activate DtC
echo -e "${GREEN}✓ Conda environment 'DtC' activated${NC}"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# Change to project directory
cd "$(dirname "$0")"
echo "  Working dir: $(pwd)"

# =============================================================================
# Phase 7: Link Module Unit Tests
# =============================================================================

if [[ "$MODE" == "all" || "$MODE" == "quick" || "$MODE" == "unit" ]]; then
    print_header "Phase 7: Link Module Unit Tests"

    cd code

    # Test 1: Module imports
    print_subheader "1. Module Import Tests"

    run_test "link module imports" "python -c '
from link import (
    MCS, get_mcs_table, register_mcs_tables_from_file,
    get_nr_cqi_table, sinr_to_cqi, cqi_to_se,
    register_bler_curves_from_file, bler_awgn_sigmoid,
    effective_sinr_eesm, eff_sinr_eesm_db, combine_eff_sinr_db,
    n_sym_per_slot, n_re_per_prb, calc_tbs_bits, re_per_prb_from_config,
    OLLA, create_olla_from_config,
    choose_mcs_from_sinr, sinr_to_se_mcs, snr_to_se_sched,
    HarqManager, HarqManagerFull,
)
print(\"All 21 APIs imported\")
'"

    run_test "backward compat: link_adapt" "python -c 'from link_adapt import MCS, choose_mcs_from_sinr, calc_tbs_bits, OLLA'"
    run_test "backward compat: csi" "python -c 'from csi import get_nr_cqi_table, sinr_to_cqi, sinr_to_se_mcs'"
    run_test "backward compat: harq" "python -c 'from harq import HarqManager, HarqManagerFull'"
    run_test "backward compat: ntn_csi" "python -c 'from ntn_csi import snr_to_se_sched'"

    # Test 2: Core functionality
    print_subheader "2. Core Functionality Tests"

    run_test "MCS tables" "python -c '
from link import get_mcs_table
t1 = get_mcs_table(\"table_1_64qam\")
t2 = get_mcs_table(\"table_2_256qam\")
t3 = get_mcs_table(\"table_3_low_se\")
assert len(t1) == 29, f\"Table 1 should have 29 entries, got {len(t1)}\"
assert len(t2) == 28, f\"Table 2 should have 28 entries, got {len(t2)}\"
assert len(t3) == 29, f\"Table 3 should have 29 entries, got {len(t3)}\"
print(f\"Tables: {len(t1)}, {len(t2)}, {len(t3)} entries\")
'"

    run_test "CQI mapping" "python -c '
import numpy as np
from link import sinr_to_cqi, cqi_to_se
sinr = np.array([-5.0, 0.0, 10.0, 20.0])
cqi = sinr_to_cqi(sinr)
se = cqi_to_se(cqi)
assert cqi[0] < cqi[-1], \"CQI should increase with SINR\"
assert se[0] < se[-1], \"SE should increase with CQI\"
print(f\"CQI range: {cqi[0]}-{cqi[-1]}, SE range: {se[0]:.2f}-{se[-1]:.2f}\")
'"

    run_test "EESM calculation" "python -c '
import numpy as np
from link import eff_sinr_eesm_db
sinr = np.array([10.0, 12.0, 8.0, 15.0])
eff = eff_sinr_eesm_db(sinr, beta_db=1.5)
assert eff < np.mean(sinr), \"EESM should be less than arithmetic mean\"
print(f\"Mean: {np.mean(sinr):.2f} dB, EESM: {eff:.2f} dB\")
'"

    run_test "TBS calculation" "python -c '
from link import get_mcs_table, calc_tbs_bits
table = get_mcs_table(\"table_1_64qam\")
tbs = calc_tbs_bits(n_prb=50, mcs=table[15])
assert tbs > 0, \"TBS should be positive\"
assert tbs < 100000, \"TBS should be reasonable\"
print(f\"MCS 15 @ 50 PRBs: {tbs} bits\")
'"

    run_test "OLLA adaptation" "python -c '
from link import OLLA
olla = OLLA(step_up_db=0.1, step_down_db=0.5)
initial = olla.offset_db
olla.update(is_ack=True)
after_ack = olla.offset_db
olla.update(is_ack=False)
after_nack = olla.offset_db
assert after_ack > initial, \"Offset should increase after ACK\"
assert after_nack < after_ack, \"Offset should decrease after NACK\"
print(f\"Offsets: {initial:.2f} -> {after_ack:.2f} -> {after_nack:.2f} dB\")
'"

    run_test "MCS selection" "python -c '
from link import choose_mcs_from_sinr
mcs_low = choose_mcs_from_sinr(5.0, \"table_1_64qam\")
mcs_high = choose_mcs_from_sinr(20.0, \"table_1_64qam\")
assert mcs_high.se > mcs_low.se, \"Higher SINR should yield higher SE MCS\"
print(f\"SINR 5dB: MCS {mcs_low.idx}, SINR 20dB: MCS {mcs_high.idx}\")
'"

    run_test "HARQ manager" "python -c '
import numpy as np
from link import HarqManager
harq = HarqManager(num_ue=4, num_procs=8, ack_delay_ttis=4)
assert harq.can_schedule(0), \"Should be able to schedule initially\"
harq.on_scheduled(np.array([0, 1, 2, 3]))
harq.advance_time()
print(\"HARQ manager works correctly\")
'"

    # Test 3: Integration with other modules
    print_subheader "3. Module Integration Tests"

    run_test "scheduler imports" "python -c '
from scheduler.baseline import pf_schedule_baseline
from scheduler.radiomap import pf_schedule_radiomap_blocks
print(\"Scheduler modules OK\")
'"

    run_test "simulation engine imports" "python -c '
from simulation.engine import SimulationEngine
from simulation.constellation_engine import ConstellationEngine
print(\"Simulation engines OK\")
'"

    run_test "core capacity imports" "python -c '
from core.capacity import se_from_snr, SEMapper
print(\"Core capacity OK\")
'"

    cd ..
fi

# =============================================================================
# Simulation Scenario Tests
# =============================================================================

if [[ "$MODE" == "all" || "$MODE" == "scenarios" ]]; then
    print_header "Simulation Scenario Tests"

    # Full scenario list
    scenarios=(
        "toronto_single"
        "toronto_constellation"
        "shanghai_single"
        "shanghai_constellation"
        "toronto_125m"
        "toronto_150m"
        "shanghai_125m"
        "shanghai_150m"
    )

    total=${#scenarios[@]}
    current=0

    for scenario in "${scenarios[@]}"; do
        ((current++))
        print_subheader "Scenario [$current/$total]: $scenario"

        echo "  Config: test/config_${scenario}.py"
        start_time=$(date +%s)

        if python run_test.py --scenario "$scenario"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo -e "  ${GREEN}✓ PASSED${NC} (${duration}s)"
            ((PASSED++))
        else
            echo -e "  ${RED}✗ FAILED${NC}"
            ((FAILED++))
        fi
    done

elif [[ "$MODE" == "quick" ]]; then
    print_header "Quick Scenario Tests (2 scenarios)"

    # Quick mode: only run 2 representative scenarios
    scenarios=(
        "shanghai_single"
        "toronto_single"
    )

    total=${#scenarios[@]}
    current=0

    for scenario in "${scenarios[@]}"; do
        ((current++))
        print_subheader "Scenario [$current/$total]: $scenario"

        echo "  Config: test/config_${scenario}.py"
        start_time=$(date +%s)

        if python run_test.py --scenario "$scenario"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo -e "  ${GREEN}✓ PASSED${NC} (${duration}s)"
            ((PASSED++))
        else
            echo -e "  ${RED}✗ FAILED${NC}"
            ((FAILED++))
        fi
    done
fi

# =============================================================================
# Summary
# =============================================================================

print_header "Test Summary"

echo ""
echo "Results:"
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✓✓✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}✗✗✗ SOME TESTS FAILED${NC}"
    echo -e "${RED}======================================================================${NC}"
    exit 1
fi
