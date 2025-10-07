"""
Single-satellite (Starlink) config centered on Toronto, Canada.
"""

CONFIG = {
    # Area of interest: Toronto
    "ref_lat_deg": 43.69069,
    "ref_lon_deg": -79.37107,

    # Use Toronto Radio Map
    "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
    "radio_map_mat_var": "XdB_recon_tensor",
    "radio_map_units": "dBm",

    # Enable single-satellite dynamic orbit
    "enable_orbit_dynamics": True,
    # Pick one Starlink satellite from tles/starlink_DTC_tle.txt
    "tle_name": "STARLINK-11090 [DTC]",
    "tle_lines": [
        "1 59422C 24065B   25266.77548611  .00029064  00000+0  23954-3 0  2662",
        "2 59422  53.1572 196.6800 0001379  82.5466  65.8632 15.69667376    15",
    ],
    # Start time near a visible pass window (UTC)
    "orbit_start_datetime": "2025-10-07T21:38:31.574982+00:00",
    
    # Disable constellation mode (use single satellite)
    "enable_constellation": False,
    "tle_catalog_path": None,

    # Reporting
    "write_json_report": True,
    "report_basename": "toronto_single",
}

