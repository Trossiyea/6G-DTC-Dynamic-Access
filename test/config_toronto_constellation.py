"""
Constellation config for Toronto using Starlink catalog.
"""

CONFIG = {
    # Area of interest: Toronto
    "ref_lat_deg": 43.69069,
    "ref_lon_deg": -79.37107,

    # Use Toronto Radio Map
    "radio_map_mat_path": "radio_map/Toronto/RadioMap/RM_toronto125_dBm.mat",
    "radio_map_mat_var": "XdB_recon_tensor",
    "radio_map_units": "dBm",

    # Constellation mode
    "enable_constellation": True,
    "tle_catalog_path": "tles/starlink_DTC_tle.txt",
    # Start time (UTC) for the scenario
    "orbit_start_datetime": "2025-09-23T12:00:00Z",

    # Reporting
    "write_json_report": True,
    "report_basename": "toronto_constellation",
}

