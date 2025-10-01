def overrides():
    return {
        # Mode
        'enable_constellation': True,
        # Geography (Calgary)
        'ref_lat_deg': 51.02356,
        'ref_lon_deg': -114.08272,
        # TLE catalog should include DTC/Starlink entries (base CONFIG points to tles/starlink_DTC_tle.txt)
        'enable_time_varying': True,
        'write_json_report': True,
        'report_basename': 'constellation_summary',
        'plot_dir': 'test/out',
    }

