#!/usr/bin/env python3
"""
Trace export tests (UI-ready per-TTI series).

These tests focus on:
- Scheduler per-TTI throughput recording (scheduled vs ACKed in HARQ mode)
- SimulationEngine/ConstellationEngine attaching `trace` outputs when enabled
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))


class TestTraceExport(unittest.TestCase):
    def test_pf_schedule_radiomap_blocks_records_harq_thr(self):
        """pf_schedule_radiomap_blocks should record scheduled/ACKed throughput in HARQ full mode."""
        from scheduler.radiomap import pf_schedule_radiomap_blocks
        from link import HarqManagerFull

        np.random.seed(0)

        n_ue, z, t = 3, 9, 6
        snr_lin = np.full((n_ue, z), 1e3, dtype=float)  # ~30 dB, near-certain ACK
        cap = np.log2(1.0 + snr_lin)

        cfg = {
            "show_progress": False,
            "mcs_table_kind": "table_1_64qam",
            "csi_mcs_table": "table_1_64qam",
            "harq_target_bler": 0.01,
            "bler_slope_db": 1.0,
            "bler_margin_db": 1.0,
            "pdsch_dmrs_sym_per_slot": 1,
            "dmrs_re_per_sym_per_prb": 6,
            "oh_prb": 0,
        }

        harq = HarqManagerFull(num_ue=n_ue, num_procs=8, ack_delay_ttis=2, config=cfg)

        ue_thr = []
        ue_ack = []

        _ = pf_schedule_radiomap_blocks(
            cap=cap,
            T=t,
            beta=0.1,
            snr_lin=snr_lin,
            overhead_eff=1.0,
            use_mcs=False,
            power_split=False,
            se_metric_override=None,
            max_prbs_per_ue=6,
            mcs_params=None,
            se_metric_time=None,
            snr_lin_time=None,
            require_contiguous=True,
            rng=np.random.default_rng(1),
            harq_mgr=harq,
            record_ue_thr=True,
            ue_thr_out=ue_thr,
            record_ue_ack_thr=True,
            ue_ack_thr_out=ue_ack,
            config=cfg,
        )

        self.assertEqual(len(ue_thr), t)
        self.assertEqual(len(ue_ack), t)

        thr0 = np.asarray(ue_thr[0], dtype=float)
        ack0 = np.asarray(ue_ack[0], dtype=float)
        self.assertEqual(thr0.shape, (n_ue,))
        self.assertEqual(ack0.shape, (n_ue,))
        self.assertTrue(np.all(thr0 >= 0.0))
        self.assertTrue(np.all(ack0 >= 0.0))

    def test_simulation_engine_attaches_trace(self):
        """SimulationEngine should attach `trace` when enable_trace is set."""
        from simulation.engine import SimulationEngine

        config = {
            "Z": 51,
            "N_UE": 4,
            "T": 8,
            "seed": 7,

            "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
            "radio_map_mat_var": "XdB_recon_tensor",
            "radio_map_units": "dBm",

            "enable_time_varying": False,
            "enable_orbit_dynamics": False,
            "enable_trace": True,
            "trace_level": "ui",

            "pf_beta": 0.1,
            "shadow_std_db": 3.0,
            "P_tx_dbm": 30.0,
            "scs_khz": 30,
            "sat_altitude_km": 600.0,
            "carrier_freq_GHz": 2.0,
            "cell_size_km": 0.125,
            "ref_lat_deg": 43.65108,
            "ref_lon_deg": -79.34702,

            "enable_harq_full": False,
            "enable_harq_deferral": False,
            "write_json_report": False,
            "show_progress": False,
        }

        out = SimulationEngine(config).run()
        self.assertIn("trace", out)
        trace = out["trace"]
        self.assertEqual(trace.get("mode"), "single")
        self.assertIn("baseline", trace)
        self.assertIn("radiomap", trace)
        self.assertEqual(len(trace.get("tti", [])), int(config["T"]))

    def test_constellation_engine_attaches_trace(self):
        """ConstellationEngine should attach `trace` when enable_trace is set."""
        from simulation.constellation_engine import ConstellationEngine

        tle_lines = [
            "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
            "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
        ]

        with tempfile.TemporaryDirectory() as td:
            cat_path = os.path.join(td, "mini.tle")
            with open(cat_path, "w", encoding="utf-8") as f:
                f.write("SAT-A\n")
                f.write(tle_lines[0] + "\n")
                f.write(tle_lines[1] + "\n")
                f.write("SAT-B\n")
                f.write(tle_lines[0] + "\n")
                f.write(tle_lines[1] + "\n")

            config = {
                "Z": 51,
                "N_UE": 3,
                "T": 4,
                "seed": 3,

                "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
                "radio_map_mat_var": "XdB_recon_tensor",
                "radio_map_units": "dBm",

                "enable_constellation": True,
                "tle_catalog_path": cat_path,
                "orbit_start_datetime": "2025-10-07T21:38:31.574982+00:00",
                "ref_lat_deg": 43.65108,
                "ref_lon_deg": -79.34702,
                "constellation_max_ground_radius_km": 40000.0,
                "min_elev_deg": -90.0,

                "enable_trace": True,
                "trace_level": "ui",

                "pf_beta": 0.1,
                "shadow_std_db": 3.0,
                "P_tx_dbm": 30.0,
                "scs_khz": 30,
                "sat_altitude_km": 600.0,
                "carrier_freq_GHz": 2.0,
                "cell_size_km": 0.125,

                "enable_harq_full": False,
                "write_json_report": False,
                "show_progress": False,
            }

            out = ConstellationEngine(config).run()
            self.assertIn("trace", out)
            trace = out["trace"]
            self.assertEqual(trace.get("mode"), "constellation")
            self.assertEqual(len(trace.get("tti", [])), int(config["T"]))
            self.assertIn("extra", trace)
            extra = trace["extra"]
            self.assertIn("candidate_sats", extra)
            self.assertEqual(len(extra.get("candidate_sats", [])), int(config["T"]))

    def test_oals_fallback_without_orbit_dynamics(self):
        """OALS should still run (StaticOrbitModel fallback) when orbit dynamics is disabled."""
        from simulation.engine import SimulationEngine

        config = {
            "Z": 51,
            "N_UE": 4,
            "T": 6,
            "seed": 11,

            "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
            "radio_map_mat_var": "XdB_recon_tensor",
            "radio_map_units": "dBm",

            "enable_time_varying": True,
            "enable_orbit_dynamics": False,
            "enable_oals": True,

            "enable_trace": True,
            "trace_level": "ui",

            "pf_beta": 0.1,
            "shadow_std_db": 3.0,
            "P_tx_dbm": 30.0,
            "scs_khz": 30,
            "sat_altitude_km": 600.0,
            "carrier_freq_GHz": 2.0,
            "cell_size_km": 0.125,
            "ref_lat_deg": 43.65108,
            "ref_lon_deg": -79.34702,

            "lookahead_horizon_ttis": 20,
            "lookahead_sample_interval": 5,
            "lookahead_update_interval": 10,
            "alpha_urgent": 0.8,
            "beta_wait": 0.3,
            "theta_lookahead": 0.7,
            "gamma_decay": 2.0,
            "boost_factor": 2.0,

            "enable_harq_full": False,
            "enable_harq_deferral": False,
            "write_json_report": False,
            "show_progress": False,
        }

        out = SimulationEngine(config).run()
        self.assertIn("trace", out)
        trace = out["trace"]
        self.assertIn("oals", trace)
        self.assertEqual(np.asarray(trace["oals"]["phi"]).shape, (int(config["T"]), int(config["N_UE"])))
        self.assertIn("oals_stats", out)
        self.assertGreaterEqual(int(out["oals_stats"].get("updates", 0)), 1)


if __name__ == "__main__":
    unittest.main()
