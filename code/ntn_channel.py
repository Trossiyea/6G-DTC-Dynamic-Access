"""
NTN channel models aligned with 3GPP TR 38.811/38.821.

DEPRECATED: This module has been refactored into code/ntn/.
This file provides backward compatibility by re-exporting from the new location.

New code should import from 'ntn' directly:
    from ntn import sample_3gpp_ntn_fading, NTNChannelProfile, describe_profile
"""

# Re-export all public APIs from ntn module for backward compatibility
from ntn.channel import (
    NTNChannelProfile,
    NTN_CHANNEL_PROFILES,
    sample_3gpp_ntn_fading,
    describe_profile,
)


__all__ = [
    "NTNChannelProfile",
    "NTN_CHANNEL_PROFILES",
    "sample_3gpp_ntn_fading",
    "describe_profile",
]
