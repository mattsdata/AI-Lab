# ==================================================================================================
# 📜 ailab-heritage
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/src/ailab_heritage/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab heritage framework: landscape archaeology from elevation data.
#    Finds candidate earthworks, records evidence with a conservative status, protects
#    unexcavated locations, and estimates how features fade over centuries (ADR-0007).
#
#  🔧 WHAT IT DOES
#    Re-exports candidate detection, evidence records, protection and landscape change.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""ailab-heritage: candidate detection, evidence, site protection and landscape change."""

# ========================================
# 📂 Imports
# ========================================
from ailab_heritage.candidates import Candidate, detect_candidates
from ailab_heritage.evidence import Evidence, SiteRecord
from ailab_heritage.landscape import FeatureFuture, degradation
from ailab_heritage.protection import REPORTING_GUIDANCE, generalise_location, public_record

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "REPORTING_GUIDANCE",
    "Candidate",
    "Evidence",
    "FeatureFuture",
    "SiteRecord",
    "degradation",
    "detect_candidates",
    "generalise_location",
    "public_record",
]
