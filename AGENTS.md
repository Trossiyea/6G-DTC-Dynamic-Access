# Repository Guidelines

## Project Structure & Module Organization
- `code/`: core simulator, organized into sub-packages:
  - `core/`: foundational utilities
    - `units.py` - unit conversions (dBm↔mW, thermal noise calculation)
    - `capacity.py` - capacity/SE calculations, MCS mapping, EESM, Shannon formulas
  - `data_io/`: data input/output
    - `radiomap.py` - Radio Map loading (MAT/HDF5 formats, auto-detection)
  - `scheduler/`: scheduling algorithms
    - `baseline.py` - 3GPP-like wideband PF scheduler
    - `radiomap.py` - RadioMap-aware contiguous block scheduler (greedy marginal ΔSE)
    - `subband.py` - subband-level baseline scheduler
    - `power_alloc.py` - DL power allocation (water-filling with box constraints)
  - `config/`: type-safe configuration system
    - `schema.py` - 15 dataclass config groups (simulation, radio_map, harq, etc.) with metadata
    - `compat.py` - ConfigDict wrapper for backward-compatible dict-style access
    - `loader.py` - JSON/YAML/Python file config loader with merge logic
    - `__init__.py` - exports CONFIG instance, load_scenario_config, and public API
  - `simulation/`: simulation engine framework (Phase 5 refactoring)
    - `state.py` - serializable state containers (SimulationState, ConstellationState) for Web API
    - `callbacks.py` - callback system (ProgressCallback, TTIMetrics, CallbackManager) for real-time updates
    - `helpers.py` - helper functions (UE position generation, noise/power control, capacity computation)
    - `engine.py` - single-satellite engine (SimulationEngine) encapsulating run_once() logic
    - `constellation_engine.py` - multi-satellite engine (ConstellationEngine) with TTI loop and handover
    - `__init__.py` - exports public API (backward-compatible run_once/run_constellation wrappers)
  - `link/`: unified link layer module (Phase 7 refactoring, 1492 lines)
    - `mcs.py` - MCS table management (3GPP TS 38.214, 3 built-in tables + external registration)
    - `cqi.py` - CQI tables and SINR→CQI→SE mapping (38.214-compliant thresholds)
    - `bler.py` - BLER curve management (sigmoid model + external curve registration)
    - `eesm.py` - EESM effective SINR calculation and HARQ soft combining
    - `tbs.py` - TBS calculation and RE statistics (TS 38.214 §5.1.3.2)
    - `olla.py` - OLLA adaptive SINR offset control (ACK/NACK feedback-driven)
    - `adaptation.py` - unified link adaptation interface (MCS selection from SINR)
    - `harq.py` - HARQ managers (simple and full modes, process management, RV cycling)
    - `__init__.py` - exports 21 public APIs
  - Main modules: `main.py` (lightweight orchestration, 274 lines), `orbit.py`, `constellation.py`, `ntn_channel.py`, `logging_utils.py`, `result_schema.py`.
  - Backward-compatible wrappers: `link_adapt.py`, `harq.py`, `csi.py`, `ntn_csi.py` (re-export from `link/` for legacy imports).
  - `__init__.py` files expose public API; scenario overrides live in `test/`.
- `test/`: city/constellation presets (`config_*.py`).
- `docs/`: reference MCS tables and templates.
- `radio_map/`: packaged MAT/NPY radio maps and converters; `tles/` holds orbit inputs.
- `tools/`: plotting/KPI helpers (e.g., `find_best_satellite.py`, `diagnose_zero_se.py`, visualization scripts); `tools/archive/` contains legacy scripts for reference. Keep outputs in `output/` or `results/` instead of committing them.

## Build, Test, and Development Commands
- Environment: `conda create --name ntn-dl --file environment.yml`; set `export MPLCONFIGDIR=$(mktemp -d)` if permissions warn.
- Scenario runs: `make help`, `make list`, `make test-toronto-single`, `make test-shanghai-constellation`, `make test-all` (wraps `python run_test.py`).
- Direct scripts: `python run_test.py --scenario toronto_single`, `python run_test.py --all`, `python run_resolution_comparison.py --city toronto`, `python code/main.py` (default config).
- Batch: `./run_all_tests.sh` exercises the shipped scenarios; prefer before large refactors.

## Coding Style & Naming Conventions
- Python, 4-space indent, type hints where possible; keep functions small and vectorized (NumPy-first). Stick to snake_case and reuse existing config key patterns (e.g., `rm_flicker_db_std`).
- Import patterns:
  - From sub-packages: `from core.units import dbm_to_mw`, `from scheduler.radiomap import pf_schedule_radiomap_blocks`, `from data_io.radiomap import load_radio_map_from_mat`
  - Configuration: `from code.config import CONFIG, load_scenario_config` (supports both dict-style `CONFIG["key"]` and typed `CONFIG.simulation.N_UE`)
  - Simulation engines: `from simulation import SimulationEngine, ConstellationEngine, SimulationState` (Phase 5 new API)
  - Link layer (Phase 7): `from link import MCS, choose_mcs_from_sinr, HarqManager, HarqManagerFull, OLLA, eff_sinr_eesm_db, calc_tbs_bits` (21 APIs available)
  - Backward-compatible: `from link_adapt import MCS, OLLA`, `from csi import sinr_to_cqi`, `from harq import HarqManager` (legacy imports still work)
  - Main entry: `from code.main import run_once, run_constellation` (legacy-compatible wrappers)
- Add docstrings for public helpers and clarify tricky math with brief comments; avoid noisy prints except for CLI progress.
- Keep data paths relative to repo root; prefer `pathlib.Path` for new utilities.

## Testing Guidelines
- Environment verification: run `python run_test.py --verify` to check Phase 1-7 module availability, dependencies, and imports (8 checks).
- Scenario validation: run a city preset after behavioral changes (`python run_test.py --scenario toronto_single`); for orbit/scheduler edits, add a constellation case.
- Unit tests: `./run_all_tests.sh --unit` runs Phase 7 link module tests (15 tests covering imports, MCS/CQI/EESM/HARQ functionality, and integration).
- Quick validation: `./run_all_tests.sh --quick` runs unit tests + 2 representative scenarios.
- Test script (`run_test.py`) uses the new `load_scenario_config()` API to automatically merge scenario overrides with default config.
- When adding features, document required data under `test/` and include sample usage in config files.

## Commit & Pull Request Guidelines
- Commit messages follow the short, action-first style seen in history (e.g., `refactor: 模块化重构 Phase 4 - 配置管理系统重构`, `docs: 更新 README.md 和 AGENTS.md`); keep under ~72 chars, optionally scoped (`orbit: tune Doppler clamp`, `config: add new parameter`).
- PRs should note what changed, why, and which commands were run (include sample output/metrics). Link relevant docs or scenarios and flag new data dependencies or expected outputs in `output/`/`results/`.

## Security & Configuration Tips
- Radio map and TLE files are large; avoid duplicating them. Never commit local caches, plots, or temporary `__pycache__`/`.ipynb_checkpoints`.
- Configuration files support three access patterns: dict-style (`CONFIG["key"]`), typed (`CONFIG.simulation.N_UE`), or file loading (`load_scenario_config("path")`).
- Validate paths in configs before running (run `make verify` once if you add scenarios). Keep secrets out of configs; TLE catalogs here are public.
