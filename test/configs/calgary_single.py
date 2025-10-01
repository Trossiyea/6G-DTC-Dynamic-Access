def overrides():
    return {
        # Mode
        'enable_constellation': False,
        'enable_orbit_dynamics': True,
        # Geography (Calgary)
        'ref_lat_deg': 51.02356,
        'ref_lon_deg': -114.08272,
        # If you have DTC TLE lines, you can override 'tle_lines' here; otherwise use base CONFIG
        'enable_time_varying': True,
        'write_json_report': False,
    }

