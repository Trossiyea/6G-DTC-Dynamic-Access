"""
Single-satellite (Satnet) config centered on Shanghai.

Select one satellite from `tles/Satnet_DTC.txt` and fix an orbit start
time so the satellite is visible with high elevation during the run.
"""

CONFIG = {
    # Area of interest: Shanghai
    "ref_lat_deg": 31.2304,
    "ref_lon_deg": 121.4737,

    # Use Shanghai Radio Map
    "radio_map_mat_path": "radio_map/Shanghai/RadioMap/RM_shanghai125_dBm.mat",
    "radio_map_mat_var": "XdB_recon_tensor",
    "radio_map_units": "dBm",

    # Enable single-satellite dynamic orbit
    "enable_orbit_dynamics": True,
    # Pick one Satnet satellite (from tles/Satnet_DTC.txt)
    "tle_name": "SATNET-590-00004 [DTC]",
    "tle_lines": [
        "1     3U 00000A   25274.57027601 .00000000  00000-0 0 000           3",
        "2     3  85.0000   0.0000 0000000   0.0000  24.0000 14.92546055    05",
    ],
    # Start time near a visible pass window (UTC)
    "orbit_start_datetime": "2025-10-08T02:19:16.314794+00:00",
    
    # Disable constellation mode (use single satellite)
    "enable_constellation": False,
    "tle_catalog_path": None,

    # Keep other behaviors from main config (scheduler, HARQ, etc.)
    # Optionally shorten/extend the sim window here if needed
    # "T": 2000,

    # Reporting
    "write_json_report": True,
    "report_basename": "shanghai_single",
}

