"""
CSI/CQI handling for scheduler metrics.

Provides an optional path to quantize instantaneous SINR to CQI and back to SE,
emulating standardized CQI reporting rather than using raw SINR->SE.
"""

from typing import Optional, Dict
import numpy as np
from csi import sinr_to_cqi, cqi_to_se


def snr_to_se_sched(snr_lin: np.ndarray,
                    use_mcs: bool,
                    mcs_params: Optional[Dict],
                    enable_cqi_quant: bool = False,
                    cqi_table: str = "nr_64qam") -> np.ndarray:
    """
    Map instantaneous SNR (linear) to SE for scheduling metric.
    - If enable_cqi_quant: quantize SINR to CQI and map to SE via table.
    - Else: if use_mcs, caller should use main.se_from_snr; here we only handle CQI path to decouple.
    """
    s = np.asarray(snr_lin, dtype=float)
    if enable_cqi_quant:
        sinr_db = 10.0 * np.log10(np.maximum(s, 1e-12))
        cqi = sinr_to_cqi(sinr_db, table=cqi_table)
        return cqi_to_se(cqi, table=cqi_table)
    # Fallback: Shannon as neutral default if not using quantization here
    return np.log2(1.0 + np.maximum(s, 0.0))

