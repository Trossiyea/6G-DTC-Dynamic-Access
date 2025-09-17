NR-NTN Radio Map–Aware Downlink Simulator
=========================================

This repository hosts a self-contained simulator for the NR non-terrestrial
network (NTN) downlink with a focus on direct-to-satellite access. It models a
radio-map-aware scheduler, realistic orbit and beam geometry, per-PRB link
adaptation, and (optionally) HARQ with OLLA. The code base has been trimmed to
downlink-only mechanics so it can be used as a light-weight analysis harness or
as a reference implementation for radio-map-driven NTN research.


Key Capabilities
----------------
- **Radio-map aware proportional fair scheduling** with contiguous PRB blocks,
  optional water-filling power allocation, and time-varying interference maps.
- **3GPP-aligned link adaptation**: CQI and spectral efficiency curves follow
  TS 38.214; default MCS tables load from `docs/mcs_tables_38_214.json`. BLER is
  calibrated around the 10% AWGN anchors and can be refined with external
  curves.
- **Orbit & beam dynamics** via a configurable `OrbitModel` (simple kinematics
  or Skyfield/TLE when available). The simulator tracks slant range, Doppler,
  and propagation delay per UE.
- **HARQ (optional)** with process management, soft combining (EESM), OLLA, and
  delayed ACK gating. Disable it for the lighter Stage-1 style analyses.
- **Extensive configurability**: CSI periodicity and delay, interference map
  estimation error, Doppler residuals, scheduler realism knobs, and plotting /
  reporting switches.


Prerequisites
-------------
- Python 3.9+ with NumPy, SciPy, Matplotlib, and (optionally) Skyfield/sgp4 for
  TLE-driven geometry. The repository has been exercised in a Conda
  environment named `ns3env` that already bundles these dependencies.
- To avoid Matplotlib cache permission warnings on shared machines, set
  `export MPLCONFIGDIR=$(mktemp -d)` before running.


Quick Start
-----------
```bash
python code/main.py
```

The entry point `code/main.py` executes a single experiment with the default
configuration (`code/config.py`) and then runs a multi-seed sweep for summary
statistics. Plots (gain histogram, radio-map slice, wideband vs best PRB
comparison) are saved under `output/` by default.

Environment variables can override selected knobs on the fly, e.g.

```bash
MEAS_T=120 MEAS_N_UE=50 python code/main.py
```


Configuration Highlights
------------------------
All tunables live in `code/config.py`. Key groups include:

- **Radio maps & traffic**: `X/Y/Z`, `N_UE`, `radio_map_mat_path`,
  `K_interferers`, `seed`.
- **Geometry & orbit**: `enable_geometry`, `enable_orbit_dynamics`,
  `sat_ground_speed_kms`, `sat_heading_deg`, `sat_altitude_km`,
  `cell_size_km`. Skyfield support is enabled by default; provide TLE lines via
  `CONFIG['tle_lines']` or `tle_path`.
- **Link budget**: `P_tx_dbm`, `L_fs_db`, `G_rx_db`, `rx_nf_db`, `impl_loss_db`.
- **Scheduler realism**: `use_mcs`, `csi_delay_ttis`,
  `baseline_csi_delay_ttis`, `rm_csi_delay_ttis`, `power_split`,
  `sched_require_contiguous`, `sched_eesm_beta_db`, `dl_power_model`.
- **HARQ / OLLA / BLER** (optional): `enable_harq_full`, `harq_max_procs`,
  `harq_ack_delay_ttis`, `harq_target_bler`, `harq_max_retx`,
  `olla_step_up_db`, `olla_step_down_db`, `olla_init_offset_db`,
  `bler_slope_db`, `bler_margin_db`.
- **MCS tables**: defaults point to the official 38.214 JSON file via
  `mcs_3gpp_table_path` plus `mcs_table_kind='3gpp_table_1'`. You can load the
  256-QAM set (`3gpp_table_2`) or low-SE set (`3gpp_table_3`) without code
  changes. The earlier approximate tables remain available as
  `table_1_64qam`, `table_2_256qam`, and `table_3_low_se`.


Example: custom experiment
--------------------------
```python
from copy import deepcopy
from config import CONFIG
from main import run_once

cfg = deepcopy(CONFIG)
cfg.update({
    "enable_time_varying": True,
    "enable_orbit_dynamics": False,
    "mcs_table_kind": "3gpp_table_2",
    "enable_harq_full": True,
    "harq_ack_delay_ttis": 4,
    "harq_max_procs": 8,
    "harq_target_bler": 0.1,
    "bler_margin_db": 2.0,
    "N_UE": 25,
    "T": 80,
    "seed": 780,
    "save_plots": False,
    "show_plots": False,
})
result = run_once(cfg)
print("Baseline default SE:", result["avg_se_baseline_default"])
print("Radio-map SE:", result["avg_se_radiomap"])
print("Gain vs default (%):", result["improvement_vs_default_pct"])
```


Outputs
-------
`run_once` returns a dictionary with the primary metrics (average SE for both
baselines, relative gains, Jain fairness, etc.) plus optional time-series
snapshots when dynamics are enabled. When `CONFIG['write_json_report']` is
true, the summary is serialized to `output/<report_basename>.json`.


Troubleshooting
---------------
- **MCS table fails to load**: ensure `mcs_3gpp_table_path` points to a valid
  JSON file that follows the template in `docs/mcs_tables_38_214_template.json`.
- **Matplotlib cache warnings**: export `MPLCONFIGDIR` to a writable temporary
  directory.
- **Skyfield unavailable**: the simulator falls back to the simple orbit model;
  set `enable_skyfield_orbit=False` to silence warnings.


Repository Layout
-----------------
- `code/`: simulation modules (`main.py`, `orbit.py`, `link_adapt.py`,
  `harq.py`, `ntn_channel.py`, etc.).
- `docs/`: official 38.214 MCS tables and templates for customization.
- `radio_map/`: example interference maps.
- `output/`: generated reports and plots (created on demand).


Roadmap (indicative)
--------------------
- Expand orbit modelling with full ECEF frame support and richer beam patterns.
- Integrate real-world interference datasets and per-beam correlation models.
- Extend HARQ statistics and logging for multi-beam / multi-cell studies.


License
-------
This simulator is provided for research and prototyping purposes. Adapt or
extend at your discretion within your project’s licensing constraints.
