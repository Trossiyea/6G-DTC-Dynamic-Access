def overrides():
    return {
        # Mode
        'enable_constellation': True,
        # Geography (Shanghai)
        'ref_lat_deg': 31.2304,
        'ref_lon_deg': 121.4737,
        # TLE catalog (use default starlink_DTC_tle.txt)
        # Dynamics
        'enable_time_varying': True,
        # Reporting
        'write_json_report': True,
        'report_basename': 'constellation_summary',
        'plot_dir': 'test/out',
    }

