# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: NR-NTN Radio Map–Aware Downlink Simulator (direct-to-satellite)

Commands
- Environment setup (Conda explicit spec)
  - bash
    path=null
    start=null
    conda create -n ntn-dl --file environment.yml
    conda activate ntn-dl
    # Optional (avoid Matplotlib cache warnings on shared machines)
    export MPLCONFIGDIR=$(mktemp -d)

- Quick start: single run (defaults in code/config.py)
  - bash
    path=null
    start=null
    python code/main.py

- Constellation mode (default True in config)
  - Requirements are satisfied by environment.yml (skyfield, sgp4). TLE catalog is at tles/Satnet_DTC.txt.
  - To switch to single-satellite mode in-place, set enable_constellation=False in code/config.py and rerun.

- Batch experiment runner (tools)
  - bash
    path=null
    start=null
    # runs multiple seeds and prints mean gains; accepts EXP_* env overrides
    python tools/exp_runner.py
    
    # Examples of overrides (see tools/exp_runner.py for full list)
    EXP_T=120 EXP_N_UE=50 EXP_SEEDS=10 EXP_SEED_START=11 \
    EXP_BASELINE_DELAY=30 EXP_RM_DELAY=0 EXP_CQI_PERIOD=20 \
    EXP_USE_WATERFILL_RM=true EXP_WF_PTOT_DBM=50 python tools/exp_runner.py

- Formal report generation (figures + CSV in output/)
  - bash
    path=null
    start=null
    # generates plots and formal_summary.csv; accepts REP_* env overrides
    python tools/report_runner.py
    
    # Example: smaller run
    REP_N_UE=20 REP_T=80 REP_SEEDS=5 python tools/report_runner.py

- Tests
  - All feature tests (self-contained script)
    - bash
      path=null
      start=null
      python code/test_features.py
  - If pytest is available in your environment:
    - bash
      path=null
      start=null
      python -m pytest -q code/test_features.py
      # single test
      python -m pytest -q code/test_features.py::test_harq_full_basic

Notes on data files
- Radio map (MAT): default path is radio_map/combined_power_51_50_50.mat; variable X_true; units mW. The Z dimension must match the PRB count specified in config (e.g., 51 for 20 MHz @ 30 kHz SCS).
- 3GPP MCS tables: docs/mcs_tables_38_214.json (loaded automatically when provided in config).

Architecture overview
- Big-picture flow (single-satellite path)
  1) Load a 3D Radio Map R[x,y,z] in dBm from MAT; verify Z matches PRB count.
  2) Sample UE grid positions (uniform) and compute geometry: FSPL, off-axis beam gain, elevation.
  3) Compute thermal noise per PRB from SCS and noise temperature; combine interference + noise.
  4) Channel realization (3GPP NTN): large-scale state (LoS/sLoS/NLoS) + Rician small-scale per PRB.
  5) Derive per-UE per-PRB SINR and Shannon capacity; also a wideband CQI-like metric.
  6) Schedule T TTIs with two baselines and a Radio Map–aware contiguous-block scheduler; optional HARQ/OLLA.
  7) Aggregate throughput as average sum spectral efficiency per PRB; optionally record assignments/time-series; write JSON/plots.

- Core modules and responsibilities
  - code/config.py: Single source of truth for simulation knobs. Includes grid/PRB sizing, noise/SCS, link budget, CSI delay/periodicity, Radio Map estimation error/blur, beam definition, orbit/TLE reference, HARQ/OLLA/BLER settings, and constellation options. Defaults enable constellation mode and JSON reporting.
  - code/main.py: Orchestrates experiments.
    - Data I/O and preprocessing: load/select Radio Map, resample Z, generate UEs.
    - Geometry/noise: compute_geometry_and_beam(), resolve_noise_and_prb_bw().
    - Channel: compute_caps() composes link budget with NTN fading (ntn_channel.py).
    - Scheduling: pf_schedule_baseline() (wideband PF) and pf_schedule_radiomap_blocks() (contiguous PRB blocks, EESM single-MCS per block, optional DL power model incl. water-filling).
    - Time variation: build_time_variation_if_enabled() supports interference flicker/drift and orbit dynamics; can compute predicted metrics under imperfect maps.
    - HARQ (optional): integrates HarqManager/HarqManagerFull; supports ACK delays, TB crediting, simple retransmission prioritization, tail flush.
    - Reports/plots: returns a dict of KPIs; can save JSON and figures.
    - Entrypoints: run_once() (single run), run_many() (multi-seed), run_constellation() (multi-satellite with association/HO and per-satellite scheduling).
  - code/orbit.py: Single-satellite geometry with optional Skyfield/TLE-driven dynamics; also provides simple beam pattern and FSPL utilities.
  - code/constellation.py: Multi-satellite Skyfield pipeline: parse TLE catalog, filter candidates by proximity, compute per-UE geometry per sat/TTI, support association metric (e.g., snr_wb), hysteresis/TTT-based handover, and expose aggregates for reporting.
  - code/ntn_channel.py: 3GPP NTN fading model (TR 38.811/38.821). Elevation-dependent LoS/sLoS/NLoS state probabilities; per-state large-scale offsets, shadowing; Rician small-scale per PRB.
  - code/csi.py and code/link_adapt.py: CQI thresholds and NR-style MCS tables, EESM mapping, RE accounting, optional external MCS/BLER table registration.
  - tools/exp_runner.py and tools/report_runner.py: Non-interactive runners for multi-seed experiments and figure/CSV generation with environment variable overrides.

- Constellation mode specifics
  - Enabled by default (CONFIG['enable_constellation']=True). Uses tles/Satnet_DTC.txt catalog; at each TTI selects candidate satellites near the mapped area, associates UEs, and runs per-satellite scheduling independently (no inter-satellite interference modeled). Produces per-satellite KPIs, optional serving traces, and aggregate HARQ stats when enabled.

- Outputs
  - JSON (when write_json_report=True) to output/<report_basename>.json with averages, optional time series, assignments, per-UE stats, and fairness.
  - Figures via main.py multi-seed plots, or via tools/report_runner.py: gain distributions, RM waterfill vs equal, PRB assignment heatmaps, Doppler/τ time-series, per-UE SE comparisons.

Important references from README
- Capabilities: Radio Map–aware contiguous-block PF (optional water-filling), 3GPP-aligned link adaptation (TS 38.214), TLE/Skyfield orbit and beam geometry, optional full HARQ with OLLA.
- Troubleshooting:
  - Ensure docs/mcs_tables_38_214.json is present or update mcs_3gpp_table_path.
  - Set MPLCONFIGDIR to avoid Matplotlib cache permission warnings.
  - For orbit dynamics/constellation mode, provide valid TLE lines/path and set ref_lat_deg/ref_lon_deg or enable auto_ref_from_tle where appropriate.

What’s intentionally not here
- No linter/formatter is configured in this repo; there are no Makefile/pyproject/tox configs. If you need linting, add your preferred tool and config first.
- File-by-file listing is omitted; consult the code/ directory for module details.
