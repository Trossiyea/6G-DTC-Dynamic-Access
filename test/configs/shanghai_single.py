def overrides():
    return {
        # Mode
        'enable_constellation': False,
        'enable_orbit_dynamics': True,
        # Geography (Shanghai)
        'ref_lat_deg': 31.2304,
        'ref_lon_deg': 121.4737,
        # TLE (use default DTC example already in base CONFIG)
        # Radio map (keep default path)
        # Dynamics
        'enable_time_varying': True,
        # Reporting
        'write_json_report': False,  # runner writes its own JSON
    }

