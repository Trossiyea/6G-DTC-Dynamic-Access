# -*- coding: utf-8 -*-
"""Benchmark module for scheduler comparison.

Includes:
- CSI Delay Sensitivity Benchmark
- OALS (Orbit-Aware Lookahead Scheduling) Benchmark
"""

from .csi_delay import (
    BenchmarkResult,
    CSIDelayBenchmarkConfig,
    run_benchmark as run_csi_delay_benchmark,
    run_single_delay,
    print_summary as print_csi_delay_summary,
    export_results as export_csi_delay_results,
)

from .oals_benchmark import (
    OALSBenchmarkResult,
    OALSBenchmarkConfig,
    run_benchmark as run_oals_benchmark,
    run_single_oals_test,
    print_summary as print_oals_summary,
    export_results as export_oals_results,
    create_oals_base_config,
)

__all__ = [
    # CSI Delay Benchmark
    "BenchmarkResult",
    "CSIDelayBenchmarkConfig",
    "run_csi_delay_benchmark",
    "run_single_delay",
    "print_csi_delay_summary",
    "export_csi_delay_results",
    # OALS Benchmark
    "OALSBenchmarkResult",
    "OALSBenchmarkConfig",
    "run_oals_benchmark",
    "run_single_oals_test",
    "print_oals_summary",
    "export_oals_results",
    "create_oals_base_config",
]
