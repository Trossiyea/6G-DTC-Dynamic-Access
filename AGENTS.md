# Repository Guidelines

## Project Structure & Module Organization
- `code/`: core simulator (`main.py`, `config.py`, `orbit.py`, `constellation.py`, `harq.py`, `link_adapt.py`, `ntn_channel.py`, `csi.py`, `logging_utils.py`, `result_schema.py`, helpers). `__init__.py` exposes public API; `config.py` holds defaults; scenario overrides live in `test/`.
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
- Add docstrings for public helpers and clarify tricky math with brief comments; avoid noisy prints except for CLI progress.
- Keep data paths relative to repo root; prefer `pathlib.Path` for new utilities.

## Testing Guidelines
- Scenario validation: run a city preset after behavioral changes (`python run_test.py --scenario toronto_single`); for orbit/scheduler edits, add a constellation case.
- When adding features, document required data under `test/` and include sample usage in config files.

## Commit & Pull Request Guidelines
- Commit messages follow the short, action-first style seen in history (`add visual scripts（专利）`, `debug: edit visualize_results`); keep under ~72 chars, optionally scoped (`orbit: tune Doppler clamp`).
- PRs should note what changed, why, and which commands were run (include sample output/metrics). Link relevant docs or scenarios and flag new data dependencies or expected outputs in `output/`/`results/`.

## Security & Configuration Tips
- Radio map and TLE files are large; avoid duplicating them. Never commit local caches, plots, or temporary `__pycache__`/`.ipynb_checkpoints`.
- Validate paths in configs before running (run `make verify` once if you add scenarios). Keep secrets out of configs; TLE catalogs here are public.
