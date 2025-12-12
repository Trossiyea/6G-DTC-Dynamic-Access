# -*- coding: utf-8 -*-
"""
Spectral efficiency computation and SNR/SE mapping utilities.

This module provides:
- SEMapper class: A configurable mapper from SNR to spectral efficiency
- Helper functions for SE computation with power splitting and EESM
- ICI penalty calculation for Doppler/CFO impairments
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from csi import sinr_to_se_mcs, effective_sinr_eesm


@dataclass
class MCSParams:
    """Parameters for MCS-based SE mapping."""
    mcs_table: str = "legacy"
    olla_offset_db: float = 0.0
    residual_freq_hz: float = 0.0
    scs_khz: float = 30.0

    @classmethod
    def from_config(cls, config: Dict) -> "MCSParams":
        """Create MCSParams from a configuration dictionary."""
        return cls(
            mcs_table=config.get("csi_mcs_table", "legacy"),
            olla_offset_db=config.get("csi_olla_offset_db", 0.0),
            residual_freq_hz=config.get("residual_freq_hz", 0.0),
            scs_khz=config.get("scs_khz", 30.0),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for legacy function compatibility."""
        return {
            "mcs_table": self.mcs_table,
            "olla_offset_db": self.olla_offset_db,
            "residual_freq_hz": self.residual_freq_hz,
            "scs_khz": self.scs_khz,
        }


def apply_ici_penalty(
    snr_lin: np.ndarray,
    residual_freq_hz: float = 0.0,
    scs_khz: float = 30.0,
) -> np.ndarray:
    """
    Apply ICI penalty from residual frequency offset (Doppler/CFO).

    The penalty models inter-carrier interference using a first-order
    approximation based on the frequency offset relative to symbol duration.

    Args:
        snr_lin: Linear SNR values
        residual_freq_hz: Residual frequency offset in Hz
        scs_khz: Subcarrier spacing in kHz

    Returns:
        SNR with ICI penalty applied
    """
    if residual_freq_hz <= 0.0:
        return np.asarray(snr_lin)

    T_sym = 1.0 / (scs_khz * 1e3)
    ici_factor = 1.0 + (2.0 * np.pi * residual_freq_hz * T_sym) ** 2
    return np.asarray(snr_lin) / ici_factor


def _apply_ici_penalty_lin(
    snr_lin: np.ndarray,
    mcs_params: Optional[Dict],
) -> np.ndarray:
    """
    Apply residual CFO/ICI penalty on linear SNR if configured in mcs_params.

    This is a legacy wrapper for apply_ici_penalty() that accepts dict params.

    Args:
        snr_lin: Linear SNR values
        mcs_params: Dictionary with 'residual_freq_hz' and 'scs_khz' keys

    Returns:
        SNR with ICI penalty applied
    """
    if mcs_params is None:
        return np.asarray(snr_lin)

    eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
    if eps_f <= 0.0:
        return np.asarray(snr_lin)

    scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
    return apply_ici_penalty(snr_lin, eps_f, scs_khz)


def se_from_snr(
    snr_lin: np.ndarray,
    use_mcs: bool,
    mcs_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Map SNR (linear) to spectral efficiency (bits/s/Hz).

    Args:
        snr_lin: Linear SNR values
        use_mcs: If True, use MCS table mapping; else use Shannon capacity
        mcs_params: Optional dict with keys:
            - 'residual_freq_hz': For ICI penalty
            - 'scs_khz': Subcarrier spacing
            - 'olla_offset_db': OLLA offset to add to SINR
            - 'mcs_table': MCS table name

    Returns:
        Spectral efficiency in bits/s/Hz
    """
    # Apply ICI penalty if configured
    if mcs_params is not None:
        eps_f = float(mcs_params.get("residual_freq_hz", 0.0) or 0.0)
        if eps_f > 0.0:
            scs_khz = float(mcs_params.get("scs_khz", 30.0) or 30.0)
            T_sym = 1.0 / (scs_khz * 1e3)
            ici_factor = 1.0 + (2.0 * np.pi * eps_f * T_sym) ** 2
            snr_lin = np.asarray(snr_lin) / ici_factor

    if use_mcs:
        sinr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
        if mcs_params is not None:
            sinr_db = sinr_db + float(mcs_params.get("olla_offset_db", 0.0))
            table = mcs_params.get("mcs_table", "legacy")
        else:
            table = "legacy"
        return sinr_to_se_mcs(sinr_db, table=table)

    return np.log2(1.0 + np.maximum(snr_lin, 0.0))


def se_from_snr_with_split(
    snr_base: Union[float, np.ndarray],
    k_prb: int,
    use_mcs: bool,
    mcs_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Apply power-split penalty and map SNR to SE.

    When transmit power is split across k PRBs, the effective SNR per PRB
    is reduced by factor k.

    Args:
        snr_base: Base SNR before power split
        k_prb: Number of PRBs sharing power
        use_mcs: If True, use MCS mapping
        mcs_params: MCS parameters dict

    Returns:
        Spectral efficiency after power split
    """
    k = max(1, int(k_prb))
    return se_from_snr(np.asarray(snr_base) / float(k), use_mcs, mcs_params)


def se_from_cap_shannon_with_split(
    cap_se: Union[float, np.ndarray],
    k_prb: int,
) -> np.ndarray:
    """
    Adjust Shannon SE for power split.

    Fallback when only Shannon SE is available (no SNR).
    Computes: SE' = log2(1 + (2^SE - 1) / k)

    Args:
        cap_se: Shannon capacity (SE) before split
        k_prb: Number of PRBs sharing power

    Returns:
        Adjusted spectral efficiency
    """
    k = max(1, int(k_prb))
    gamma = np.maximum(0.0, np.power(2.0, np.asarray(cap_se)) - 1.0) / float(k)
    return np.log2(1.0 + gamma)


def block_se_from_snr_vec(
    snr_lin_vec: np.ndarray,
    k_prb: int,
    use_mcs: bool,
    mcs_params: Optional[Dict],
    eesm_beta_db: float,
) -> float:
    """
    Compute per-PRB SE for a contiguous RB block using EESM.

    For MCS mode:
    1. Apply ICI penalty
    2. Divide SNR by k (power split)
    3. Compute effective SINR via EESM
    4. Map to SE via MCS table

    For Shannon mode:
    - Average log2(1 + SNR/k) across the block

    Args:
        snr_lin_vec: Linear SNR values for each PRB in block
        k_prb: Block size in PRBs (for power split)
        use_mcs: If True, use MCS mapping
        mcs_params: MCS parameters dict
        eesm_beta_db: EESM beta parameter in dB

    Returns:
        Per-PRB spectral efficiency (not block sum)
    """
    k = max(1, int(k_prb))
    s = np.asarray(snr_lin_vec, dtype=float)
    s = _apply_ici_penalty_lin(s, mcs_params)

    if use_mcs:
        sinr_db_vec = 10.0 * np.log10(np.maximum(s / float(k), 1e-12))
        sinr_eff_db = effective_sinr_eesm(sinr_db_vec, beta_db=float(eesm_beta_db), axis=-1)

        # Apply OLLA offset if present
        if mcs_params is not None:
            sinr_eff_db = sinr_eff_db + float(mcs_params.get("olla_offset_db", 0.0))

        table = mcs_params.get("mcs_table", "legacy") if mcs_params else "legacy"
        se = float(np.asarray(sinr_to_se_mcs(sinr_eff_db, table=table)))
        return se
    else:
        se_vec = np.log2(1.0 + np.maximum(s / float(k), 0.0))
        return float(np.mean(se_vec))


# Legacy alias for backward compatibility
_block_se_from_snr_vec = block_se_from_snr_vec


def se_metric_strategy(
    use_mcs: bool,
    snr_lin: Optional[np.ndarray] = None,
    cap_shannon: Optional[np.ndarray] = None,
    mcs_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Strategy for computing scheduling metric SE.

    Prefers mapping from snr_lin if provided; otherwise uses Shannon SE
    when MCS is disabled.

    Args:
        use_mcs: If True, use MCS mapping
        snr_lin: Linear SNR values (preferred)
        cap_shannon: Shannon capacity values (fallback)
        mcs_params: MCS parameters dict

    Returns:
        Spectral efficiency for scheduling decisions

    Raises:
        ValueError: If neither snr_lin nor valid cap_shannon is provided
    """
    if snr_lin is not None:
        return se_from_snr(snr_lin, use_mcs, mcs_params)
    if (not use_mcs) and (cap_shannon is not None):
        return cap_shannon
    raise ValueError("se_metric_strategy requires snr_lin or (cap_shannon with use_mcs=False)")


class SEMapper:
    """
    Configurable SNR to Spectral Efficiency mapper.

    This class encapsulates the logic for mapping SNR values to SE,
    supporting both Shannon capacity and MCS-based mapping with
    optional ICI penalty and OLLA offset.

    Example:
        >>> mapper = SEMapper.from_config(CONFIG)
        >>> se = mapper.snr_to_se(snr_lin)
        >>> block_se = mapper.block_se(snr_vec, k_prb=5, eesm_beta_db=2.5)
    """

    def __init__(
        self,
        use_mcs: bool = True,
        mcs_table: str = "legacy",
        olla_offset_db: float = 0.0,
        residual_freq_hz: float = 0.0,
        scs_khz: float = 30.0,
    ):
        """
        Initialize SEMapper.

        Args:
            use_mcs: If True, use MCS table mapping; else Shannon
            mcs_table: MCS table name (e.g., 'legacy', '3gpp_table_1')
            olla_offset_db: OLLA offset to add to SINR (dB)
            residual_freq_hz: Residual frequency offset for ICI (Hz)
            scs_khz: Subcarrier spacing (kHz)
        """
        self.use_mcs = use_mcs
        self.mcs_table = mcs_table
        self.olla_offset_db = olla_offset_db
        self.residual_freq_hz = residual_freq_hz
        self.scs_khz = scs_khz

    @classmethod
    def from_config(cls, config: Dict) -> "SEMapper":
        """
        Create SEMapper from a configuration dictionary.

        Args:
            config: Dict with keys like 'use_mcs', 'csi_mcs_table', etc.

        Returns:
            Configured SEMapper instance
        """
        return cls(
            use_mcs=config.get("use_mcs", True),
            mcs_table=config.get("csi_mcs_table", "legacy"),
            olla_offset_db=config.get("csi_olla_offset_db", 0.0),
            residual_freq_hz=config.get("residual_freq_hz", 0.0),
            scs_khz=config.get("scs_khz", 30.0),
        )

    def _get_mcs_params(self) -> Dict:
        """Get MCS params as legacy dict."""
        return {
            "mcs_table": self.mcs_table,
            "olla_offset_db": self.olla_offset_db,
            "residual_freq_hz": self.residual_freq_hz,
            "scs_khz": self.scs_khz,
        }

    def snr_to_se(self, snr_lin: np.ndarray) -> np.ndarray:
        """
        Map linear SNR to spectral efficiency.

        Args:
            snr_lin: Linear SNR values

        Returns:
            Spectral efficiency in bits/s/Hz
        """
        return se_from_snr(snr_lin, self.use_mcs, self._get_mcs_params())

    def snr_to_se_with_split(
        self,
        snr_base: Union[float, np.ndarray],
        k_prb: int,
    ) -> np.ndarray:
        """
        Map SNR to SE with power split penalty.

        Args:
            snr_base: Base SNR before split
            k_prb: Number of PRBs sharing power

        Returns:
            Spectral efficiency after power split
        """
        return se_from_snr_with_split(
            snr_base, k_prb, self.use_mcs, self._get_mcs_params()
        )

    def block_se(
        self,
        snr_lin_vec: np.ndarray,
        k_prb: int,
        eesm_beta_db: float,
        power_split: bool = False,
    ) -> float:
        """
        Compute SE for a PRB block using EESM.

        Args:
            snr_lin_vec: Linear SNR for each PRB in block
            k_prb: Block size in PRBs
            eesm_beta_db: EESM beta parameter (dB)
            power_split: If True, apply power split penalty

        Returns:
            Per-PRB spectral efficiency
        """
        k = k_prb if power_split else 1
        return block_se_from_snr_vec(
            snr_lin_vec, k, self.use_mcs, self._get_mcs_params(), eesm_beta_db
        )

    def apply_ici_penalty(self, snr_lin: np.ndarray) -> np.ndarray:
        """
        Apply ICI penalty to SNR values.

        Args:
            snr_lin: Linear SNR values

        Returns:
            SNR with ICI penalty applied
        """
        return apply_ici_penalty(snr_lin, self.residual_freq_hz, self.scs_khz)
