"""
Constellation config for Shanghai using Satnet catalog.
"""

CONFIG = {
    # Area of interest: Shanghai
    "ref_lat_deg": 31.2304,
    "ref_lon_deg": 121.4737,

    # Use Shanghai Radio Map
    "radio_map_mat_path": "radio_map/Shanghai/RadioMap/RM_shanghai125_dBm.mat",
    "radio_map_mat_var": "XdB_recon_tensor",
    "radio_map_units": "dBm",

    # Shanghai frequency band: 1890-1910 MHz, center at 1900 MHz
    "carrier_freq_GHz": 1.90,

    # Constellation mode
    "enable_constellation": True,
    "tle_catalog_path": "tles/Satnet_DTC.txt",
    # Start time (UTC) for the scenario
    "orbit_start_datetime": "2025-10-08T02:19:16.314794+00:00",
    
    # Disable single-satellite TLE (use catalog instead)
    "tle_name": None,
    "tle_lines": None,
    "tle_path": None,

    # Reporting
    "write_json_report": True,
    "report_basename": "shanghai_constellation",
}

