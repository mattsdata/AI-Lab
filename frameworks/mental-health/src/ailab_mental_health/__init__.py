# ==================================================================================================
# 📜 ailab-mental-health
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/src/ailab_mental_health/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab mental-health framework: private self-monitoring and validated
#    screening scores, built safety-first and explicitly non-diagnostic (ADR-0006).
#
#  🔧 WHAT IT DOES
#    Re-exports screening instruments, safety resources, the encrypted journal and summaries.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: PHQ-9 / GAD-7 scoring, crisis resources, encrypted journal, summary.
# ==================================================================================================
"""ailab-mental-health: private, safety-first, non-diagnostic wellbeing tools."""

# ========================================
# 📂 Imports
# ========================================
from ailab_mental_health.instruments import (
    NOT_A_DIAGNOSIS,
    ScreeningResult,
    score_gad7,
    score_phq9,
)
from ailab_mental_health.journal import CheckIn, Cipher, FernetCipher, JournalStore
from ailab_mental_health.safety import (
    CRISIS_RESOURCES,
    CrisisResource,
    mentions_crisis,
    safety_message,
)
from ailab_mental_health.summary import WellbeingSummary, wellbeing_summary

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "CRISIS_RESOURCES",
    "NOT_A_DIAGNOSIS",
    "CheckIn",
    "Cipher",
    "CrisisResource",
    "FernetCipher",
    "JournalStore",
    "ScreeningResult",
    "WellbeingSummary",
    "mentions_crisis",
    "safety_message",
    "score_gad7",
    "score_phq9",
    "wellbeing_summary",
]
