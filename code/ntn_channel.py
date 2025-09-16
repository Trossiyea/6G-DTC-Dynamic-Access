"""NTN channel models aligned with 3GPP TR 38.811/38.821.

This module provides a stochastic fading generator for the direct-to-satellite
service link. The implementation follows the three-state statistical model
defined for land-mobile UEs (LoS / shadowed-LoS / NLoS) using the parameter
sets from 3GPP TR 38.811. The state probabilities are modelled as smooth
logistic functions of the elevation angle (degrees) to match the tables in
Annex 6 of TR 38.811. Within each state we apply the additional large-scale
loss, shadowing standard deviation and Rician K-factor recommended by the
specification. The small-scale fading is i.i.d. across PRBs with unit mean
power so that long-term averages match the configured link budget.

The generator exposes a single entry point ``sample_3gpp_ntn_fading`` which
returns per-UE large-scale receive power offsets (dB), Rician fading power
gains per PRB, and the discrete state labels. The caller is responsible for
converting these to received power levels and SINR once the interference/noise
term is known.

The defaults correspond to the "S-band, handheld UE, urban" profile that is
used widely for NR-NTN studies. Alternative profiles can be added by extending
``NTN_CHANNEL_PROFILES`` or by overriding individual parameters via the
``channel_params`` dictionary.

References:
  * 3GPP TR 38.811 v17.0.0 – Annex 6 (channel model for NTN)
  * 3GPP TR 38.821 v17.0.0 – Service link modelling assumptions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


StateProbTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class NTNChannelProfile:
    """Container for 3GPP NTN fading parameters."""

    additional_loss_db: Dict[str, float]
    shadow_sigma_db: Dict[str, float]
    k_factor_db: Dict[str, float]
    # Logistic parameters (slope, mid-point) for LoS and NLoS probabilities.
    # Shadowed-LoS is derived as the remaining probability mass.
    los_logistic: Tuple[float, float]
    nlos_logistic: Tuple[float, float]


def _logistic(elev_deg: np.ndarray, slope: float, mid: float) -> np.ndarray:
    """Standard logistic curve with the angle expressed in degrees."""
    elev = np.asarray(elev_deg, dtype=float)
    return 1.0 / (1.0 + np.exp(-slope * (elev - mid)))


def _state_probabilities(elev_deg: np.ndarray, profile: NTNChannelProfile) -> StateProbTuple:
    """Return per-state probabilities for each elevation angle sample."""
    elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
    slope_los, mid_los = profile.los_logistic
    slope_nlos, mid_nlos = profile.nlos_logistic

    p_los = _logistic(elev, slope_los, mid_los)
    # For NLoS we flip the logistic so that low elevation approaches one.
    p_nlos = 1.0 / (1.0 + np.exp(slope_nlos * (elev - mid_nlos)))
    p_slos = np.maximum(0.0, 1.0 - p_los - p_nlos)

    total = np.maximum(1e-12, p_los + p_slos + p_nlos)
    p_los /= total
    p_slos /= total
    p_nlos /= total
    return p_los, p_slos, p_nlos


def _sample_states(rng: np.random.Generator,
                   p_los: np.ndarray,
                   p_slos: np.ndarray,
                   p_nlos: np.ndarray) -> np.ndarray:
    """Sample a discrete state for each UE according to the provided CDF."""
    u = rng.random(size=p_los.shape[0])
    thresh_los = p_los
    thresh_slos = p_los + p_slos
    states = np.empty(p_los.shape[0], dtype='<U5')
    states[u <= thresh_los] = 'los'
    mask_slos = (u > thresh_los) & (u <= thresh_slos)
    states[mask_slos] = 'slos'
    states[u > thresh_slos] = 'nlos'
    return states


def _rician_power_samples(rng: np.random.Generator,
                          size: Tuple[int, int],
                          k_db: float) -> np.ndarray:
    """Sample squared Rician envelope with unit mean power for a K-factor (dB)."""
    if not np.isfinite(k_db):
        k_db = -np.inf
    if k_db <= -100.0:
        k_lin = 0.0
    else:
        k_lin = 10.0 ** (k_db / 10.0)

    if k_lin == 0.0:
        sigma = np.sqrt(0.5)
        x = rng.normal(0.0, sigma, size=size)
        y = rng.normal(0.0, sigma, size=size)
        return x * x + y * y

    s = np.sqrt(k_lin / (k_lin + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
    x = rng.normal(loc=s, scale=sigma, size=size)
    y = rng.normal(loc=0.0, scale=sigma, size=size)
    return x * x + y * y


# Default profile derived from TR 38.811 Annex 6 (handheld, S-band, urban)
NTN_CHANNEL_PROFILES: Dict[str, NTNChannelProfile] = {
    "s_band_handheld_urban": NTNChannelProfile(
        additional_loss_db={
            "los": 0.0,
            "slos": 10.0,
            "nlos": 22.0,
        },
        shadow_sigma_db={
            "los": 4.0,
            "slos": 6.0,
            "nlos": 8.0,
        },
        k_factor_db={
            "los": 12.0,
            "slos": 6.0,
            "nlos": -np.inf,  # Rayleigh
        },
        # Logistic fits tuned to TR 38.811 Annex 6 (50% LoS @ ~35°, rapid NLoS drop past 20°).
        los_logistic=(0.16, 35.0),
        nlos_logistic=(0.19, 20.0),
    ),
}


def sample_3gpp_ntn_fading(
    rng: np.random.Generator,
    elevation_deg: np.ndarray,
    num_prb: int,
    profile_name: str = "s_band_handheld_urban",
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate large-scale offsets and small-scale fading per PRB.

    Returns a tuple ``(large_scale_db, fading_lin, states)`` where:

    * ``large_scale_db`` is the per-UE additive term (dB) applied on top of
      the deterministic link budget (positive boosts, negative attenuation).
    * ``fading_lin`` contains Rician power gains of shape [UE, PRB] with unit
      mean power.
    * ``states`` is an array of strings ``{"los", "slos", "nlos"}`` for
      diagnostic purposes.
    """

    if profile_name not in NTN_CHANNEL_PROFILES:
        raise KeyError(f"Unknown NTN channel profile '{profile_name}'.")
    profile = NTN_CHANNEL_PROFILES[profile_name]

    p_los, p_slos, p_nlos = _state_probabilities(elevation_deg, profile)
    states = _sample_states(rng, p_los, p_slos, p_nlos)

    overrides = overrides or {}
    add_loss = dict(profile.additional_loss_db)
    add_loss.update(overrides.get("additional_loss_db", {}))
    sigma_db = dict(profile.shadow_sigma_db)
    sigma_db.update(overrides.get("shadow_sigma_db", {}))
    k_db = dict(profile.k_factor_db)
    k_db.update(overrides.get("k_factor_db", {}))

    num_ue = elevation_deg.shape[0]
    large_scale_db = np.zeros(num_ue, dtype=float)
    fading_lin = np.ones((num_ue, num_prb), dtype=float)

    for state in ("los", "slos", "nlos"):
        mask = states == state
        if not np.any(mask):
            continue
        loss_db = -float(add_loss.get(state, 0.0))
        shadow_std = float(sigma_db.get(state, 0.0))
        large_scale_db[mask] = loss_db + rng.normal(0.0, shadow_std, size=np.count_nonzero(mask))
        fading_lin[mask] = _rician_power_samples(rng, (np.count_nonzero(mask), num_prb), float(k_db.get(state, -np.inf)))

    return large_scale_db, fading_lin, states


def describe_profile(profile_name: str) -> Dict[str, Dict[str, float]]:
    """Expose the raw profile parameters (useful for reports/tests)."""
    if profile_name not in NTN_CHANNEL_PROFILES:
        raise KeyError(f"Unknown NTN channel profile '{profile_name}'.")
    p = NTN_CHANNEL_PROFILES[profile_name]
    return {
        "additional_loss_db": dict(p.additional_loss_db),
        "shadow_sigma_db": dict(p.shadow_sigma_db),
        "k_factor_db": dict(p.k_factor_db),
        "los_logistic": tuple(p.los_logistic),
        "nlos_logistic": tuple(p.nlos_logistic),
    }
