Test scenarios and runners

This folder provides ready-to-run scenario configs and scripts to evaluate Baseline-Default vs Radio Map (RM) across seeds, generate JSON summaries and plots.

Scenarios
- shanghai_single: Shanghai city, single satellite (DtC)
- shanghai_constellation: Shanghai region, Starlink/DTC constellation
- calgary_single: Calgary, single satellite (DtC)
- calgary_constellation: Calgary region, Starlink/DTC constellation

Power models
- equal: both baseline and RM use equal per-PRB power
- waterfill: both baseline and RM use water-filling with total-power constraint

Outputs
- Per scenario, results are written under `test/out/<scenario>/<power>/`:
  - `summary.csv`: per-seed SE metrics and improvement
  - `summary.json`: aggregate stats (mean/std/median/P10/P90, bootstrap 95% CI)
  - `gain_histogram.png`: improvement (%) distribution over seeds
  - `se_vs_seed.png`: avg SE vs seed for Baseline-Default and RM
  - (constellation only) `constellation_summary_<seed>.json`: per-seed JSON reports

Requirements
- Python 3.9+
- numpy, scipy, matplotlib, skyfield, sgp4

Quick start
- Shanghai / Single / Equal power:
  - `bash test/run_shanghai_single_equal.sh`
- Shanghai / Constellation / Waterfill:
  - `bash test/run_shanghai_constellation_waterfill.sh`

Environment overrides
- `SEEDS` (default 20): number of seeds
- `SEED_START` (default 1): starting seed value
- `T` (default 2000): TTIs per run
- `N_UE` (default 100): #UEs

Examples
- Run Calgary constellation, equal power, fewer seeds:
  - `SEEDS=8 T=1500 N_UE=120 bash test/run_calgary_constellation_equal.sh`

Radio Map resolution sensitivity
- Script: `test/sensitivity_radiomap_res.py`
- Purpose: evaluate RM improvement across different Radio Map XY resolutions (e.g., 25x25, 35x35, 50x50).
- Usage (base MAT exists at CONFIG['radio_map_mat_path']):
  - `bash test/run_rm_resolution_shanghai_single_equal.sh`
  - `bash test/run_rm_resolution_shanghai_single_waterfill.sh`
  - `bash test/run_rm_resolution_calgary_single_equal.sh`
  - `bash test/run_rm_resolution_calgary_single_waterfill.sh`
- Custom MAT inputs:
  - Provide `--mats p1,p2,...` to skip resampling (each should be 3D MAT with var `X_true`).
  - Or use `--base-mat path/to/base.mat --sizes 25,35,50` to resample XY to requested sizes.
- Outputs under `test/out/rm_resolution/<scenario>/<power>/`:
  - `imp_vs_resolution.png`: mean gain vs resolution with error bars (std)
  - `gain_hist_<res>.png`: per-resolution gain histogram
  - `resolution_summary.csv`: mean/std by resolution
  - Per-resolution `summary.csv` with per-seed metrics
