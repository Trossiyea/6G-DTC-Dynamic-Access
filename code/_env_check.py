import sys
print('python:', sys.version)
try:
    import numpy as np
    import scipy
    import matplotlib
    print('numpy:', np.__version__)
    print('scipy:', scipy.__version__)
    print('matplotlib:', matplotlib.__version__)
except Exception as e:
    print('import error:', e)

try:
    from main import run_once
    from config import CONFIG
    from copy import deepcopy
    # quick smoke with TA/precomp enabled and orbit dynamics
    cfg = deepcopy(CONFIG)
    cfg.update({
        'enable_time_varying': True,
        'enable_orbit_dynamics': True,
        'enable_ntn_freq_precomp': True,
        'freq_precomp_update_ttis': 1,
        'freq_precomp_latency_ttis': 1,
        'freq_precomp_error_std_hz': 100.0,
        'freq_precomp_quant_hz': 10.0,
        'enable_ta_model': True,
        'ta_update_ttis': 20,
        'ta_latency_ttis': 2,
        'ta_margin_us': 0.3,
        'seed': 7,
    })
    out = run_once(cfg)
    print('smoke avg_se_baseline:', out['avg_se_baseline'])
    print('smoke avg_se_radiomap:', out['avg_se_radiomap'])
    print('tau_time shape:', None if out['tau_time'] is None else out['tau_time'].shape)
    print('fd_time shape:', None if out['fd_time'] is None else out['fd_time'].shape)
except Exception as e:
    print('smoke error:', type(e).__name__, e)

