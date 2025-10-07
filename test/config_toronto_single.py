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
    "tle_name": "STARLINK-11075 [DTC]",
    "tle_lines": [
        "1 58706C 24002B   25266.77201389  .00026832  00000+0  22113-3 0  2661",
        "2 58706  53.1569  16.9264 0001111  95.9280 143.4400 15.69671433    14",
    ],
    # Start time near a visible pass window (UTC)
    "orbit_start_datetime": "2025-10-08T09:22:45.447459+00:00",

    # Reporting
    "write_json_report": True,
    "report_basename": "toronto_single",
}

