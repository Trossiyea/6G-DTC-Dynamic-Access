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

    # Toronto frequency band: 2180-2200 MHz, center at 2190 MHz
    "carrier_freq_GHz": 2.19,

    # Constellation mode
    "enable_constellation": True,
    "tle_catalog_path": "tles/starlink_DTC_tle.txt",
    # Start time (UTC) for the scenario
    "orbit_start_datetime": "2025-10-07T21:38:31.574982+00:00",
    
    # Disable single-satellite TLE (use catalog instead)
    "tle_name": None,
    "tle_lines": None,
    "tle_path": None,

    # Reporting
    "write_json_report": True,
    "report_basename": "toronto_constellation",
}

