# Experimental Plan (IEEE TMC Target) — RadioMap-Aided DtS DL Scheduling

This document outlines the experiment suite and figures for a top-tier journal submission (e.g., IEEE TMC) for **direct-to-satellite (DtS) downlink** under **co-channel terrestrial BS interference**, comparing:
- **3GPP-like CSI/CQI-based scheduling** (with CSI delay + CQI periodicity),
- **RadioMap-aided scheduling** using (i) heuristic, (ii) MLP-scored NS-GBS, (iii) ISAB-scored NS-GBS.

All figures below are designed to have **≥2 subplots** (labeled (a), (b), …).

---

## 1) Research Questions (RQs) and Claims to Validate

**RQ1 (Main gain):** Does RadioMap-aided scheduling improve spectral efficiency (SE) vs 3GPP-like CSI/CQI scheduling in DtS DL with terrestrial co-channel interference?

**RQ2 (Why it helps):** How does the gain vary with **CSI delay/periodicity** (baseline) vs **map freshness / staleness** (RadioMap)?

**RQ3 (Robustness):** How sensitive is performance to RadioMap imperfections: estimation error, blur, interference non-stationarity (flicker), orbit-induced dynamics (Doppler/ICI)?

**RQ4 (User experience):** Does the method improve cell-edge users (e.g., 5%-tile throughput) while maintaining fairness?

**RQ5 (Practicality):** What is the runtime/compute overhead (CPU/GPU), and what is the “bang-per-FLOP” of MLP vs ISAB?

**RQ6 (Generalization):** Do learned schedulers generalize across cities (Toronto↔Shanghai), orbits/satellites, UE density, and RadioMap resolutions?

**RQ7 (Upper bounds):** How close do RadioMap methods get to an **oracle CSI** baseline (no delay, perfect CSI)?

---

## 2) Metrics (What to Report)

Minimum set for TMC-quality evaluation:
- **System SE** (avg bits/s/Hz per PRB; already produced as `avg_se_*`).
- **System throughput** (bps) and **avg UE throughput** (bps/UE).
- **User distribution metrics**: 5%-tile / 10%-tile / median per-UE throughput; CDF curves.
- **Fairness**: Jain’s index on per-UE throughput; optionally Gini coefficient.
- **Reliability**: HARQ ACK rate, avg retransmissions per acked TB, TB drop rate (if enabled).
- **Outage**: fraction of TTIs where UE not served / SINR below threshold (if tracked).
- **Complexity**: wall-clock time per simulated TTI; `score_calls`, `avg_score_ms_per_call`, total scheduler time.

Statistics requirements:
- Use **paired seeds** across methods (same scenario+seed) and report **mean ± 95% CI** (bootstrap) and/or median with IQR.
- For claims vs baseline, include a paired significance test (e.g., Wilcoxon signed-rank on per-seed means).

---

## 3) Compared Methods (Legend)

**Baselines**
- **B1: 3GPP-like CSI/CQI scheduler (practical)**: delayed CSI (`baseline_csi_delay_ttis>0`) + CQI quantization + CQI periodicity.
- **B2: Oracle CSI scheduler (upper bound)**: identical to B1 but **CSI delay = 0** and periodicity minimal (or every TTI).
- **B3: RadioMap heuristic (no learning)**: current heuristic scoring in NS-GBS framework (sanity baseline).

**Proposed**
- **P1: RadioMap + NS-GBS (MLP)**: `nsgbs_scorer.pt`.
- **P2: RadioMap + NS-GBS (ISAB)**: `nsgbs_isab_*.pt` (report best τ and sensitivity).

Recommended “controls” to keep fixed in most experiments:
- same PRB budget, power model, HARQ settings, orbit model, channel profile.
- same `(scenario, N_UE, T, seeds)` across methods.

---

## 4) Scenario Matrix (Where to Test)

Use a matrix to avoid “single-scenario” criticism:
- **Cities / RadioMaps**: Toronto + Shanghai.
- **Orbit modes**: single satellite + constellation (multi-sat association/handover).
- **RadioMap resolution**: 125 m vs 150 m (already in `test/config_*_125m.py`, `*_150m.py`).
- **UE load**: at least 3 points (e.g., 20, 50, 100 UEs).
- **Dynamics**: time-varying interference enabled + orbit dynamics enabled.

Minimum recommended evaluation set:
- 2 cities × 2 orbit modes × 3 UE loads × 5–10 seeds (depending on runtime budget).

---

## 5) Experiment Suite → Figures (≥2 Subplots Each)

### Figure 1 — Main Performance Across Scenarios
**Goal:** establish headline gains and consistency.
- (a) **Mean SE** by method across scenarios (bar/point plot with 95% CI).
- (b) **Gain vs practical 3GPP baseline (B1)** (%) across scenarios (paired per seed).

### Figure 2 — User-Level Throughput Distribution & Fairness
**Goal:** show user experience improvements (not only average).
- (a) **CDF** of per-UE throughput (baseline vs heuristic vs MLP vs ISAB) for a representative scenario.
- (b) **Jain fairness index** and **5%-tile throughput** (two y-axes or two curves) across methods.

### Figure 3 — CSI Delay vs RadioMap Freshness (Core Mechanism)
**Goal:** justify the information advantage rigorously.
- (a) Baseline SE vs **CSI delay** (`baseline_csi_delay_ttis`: 0, 5, 10, 20, 40).
- (b) RadioMap methods SE vs **map “age” / staleness** (`rm_csi_delay_ttis` and/or enabling RM periodicity).

Notes:
- Frame this as “baseline depends on CSI, RM depends on prior map updates”.
- Include an “oracle CSI” curve (B2) as an upper bound in (a).

### Figure 4 — Robustness to RadioMap Imperfections
**Goal:** show the scheme degrades gracefully.
- (a) SE vs **RadioMap estimation error** (`radiomap_est_error_db` sweep).
- (b) SE vs **RadioMap blur / spatial mismatch** (`radiomap_blur_sigma` sweep and/or resolution 125m vs 150m).

### Figure 5 — Non-Stationary Interference + Orbit Dynamics Stress Test
**Goal:** validate under fast-changing conditions.
- (a) SE vs **interference flicker std** (`rm_flicker_db_std` sweep).
- (b) SE vs **residual Doppler fraction** (`doppler_residual_fraction`), optionally with/without ICI modeling.

### Figure 6 — Scalability With UE Load and Bandwidth
**Goal:** show performance under load + scheduling hardness.
- (a) SE and 5%-tile throughput vs **N_UE** (e.g., 20/50/100/200).
- (b) SE vs **PRB count Z / bandwidth** (if you have multiple maps or can sub-sample PRBs consistently).

### Figure 7 — Constellation Mode: Association + Handover + Multi-Sat KPI
**Goal:** show relevance beyond single-satellite toy setups.
- (a) SE and throughput vs **#candidate sats per TTI** (`constellation_max_sats_per_tti`) and/or **min elevation**.
- (b) **HO events / outage** vs method; optionally per-satellite load distribution (served UE count).

### Figure 8 — Reliability & HARQ Goodput
**Goal:** ensure gains aren’t coming from unrealistic BLER/ACK artifacts.
- (a) **ACK rate** and **TB drop rate** across methods.
- (b) **Avg retransmissions per acked TB** and **OLLA offset evolution** (mean over time).

### Figure 9 — Runtime / Complexity / Deployability
**Goal:** satisfy “practicality” review criteria.
- (a) **Scheduler compute time per TTI** (or per run) by method (CPU vs GPU if available).
- (b) **Model scoring overhead**: `score_calls`, `avg_score_ms_per_call`, and achieved SE (scatter plot “SE vs cost”).

### Figure 10 — Learning Generalization + Ablations
**Goal:** preempt “overfitting” concerns and explain what matters.
- (a) **Cross-city generalization**: train Toronto → test Shanghai, and train Shanghai → test Toronto (gain vs baseline).
- (b) **Model ablations**: MLP vs ISAB; ISAB τ sweep; feature toggles (`add_z`, `add_step`, HARQ features).

---

## 6) Tables (Recommended for TMC)

- **Table I (System Parameters):** carrier freq, SCS, bandwidth/PRBs, channel model/profile, HARQ parameters, CSI delay/periodicity, power model, UE density, satellite orbit config.
- **Table II (Summary Results):** mean SE and gain (%) across the full scenario matrix, with CI.
- **Table III (Complexity):** runtime per TTI, memory footprint, model size (params), inference device.

---

## 7) Reproducibility Checklist (What to Log)

For each run (scenario, seed, method), log:
- All config knobs that affect realism: CSI delay/periodicity, RM error/blur, orbit dynamics, HARQ, power model.
- Model identifiers: checkpoint path + meta JSON hash (feature_dim, add_z/add_step, τ).
- Output CSV with per-run metrics + a consolidated summary CSV for each sweep.

Recommended artifacts:
- A single script (or Makefile targets) that reproduces each figure sweep and writes a `results/*.csv` used by plotting code.

---

## 8) Mapping to This Repo (Where to Run)

Primary runner for method comparison (heuristic/MLP/ISAB) across scenarios/seeds:
- `tools/run_nsgbs_bench.py` (writes `output/nsgbs_bench.csv`)

Scenario configs:
- `test/config_*` (Toronto/Shanghai, single/constellation, 125m/150m)

Training:
- `train_nsgbs.py` (MLP, outputs `output/models/nsgbs_scorer.pt` + `.json`)
- `train_nsgbs_isab.py` (ISAB, outputs `output/models/nsgbs_isab_*.pt` + `.json`)

Sweeps for Figures 3–6/8–10 can be implemented as additional “runner” scripts under `tools/` that:
1) build a config grid,
2) call `main.run_once()` (or constellation),
3) dump CSV (one per figure),
4) plot.

