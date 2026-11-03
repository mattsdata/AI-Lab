# ==================================================================================================
# 📜 ailab-mental-health :: instruments
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/src/ailab_mental_health/instruments.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Score two widely used, validated screening questionnaires exactly as published, so that
#    a self-monitoring app or a research pipeline never invents its own cut-offs: the PHQ-9
#    (low mood) and the GAD-7 (anxiety).
#
#  🔧 WHAT IT DOES
#    - score_phq9 : 9 answers (0-3) -> total 0-27, severity band, safety alert on item 9.
#    - score_gad7 : 7 answers (0-3) -> total 0-21, severity band.
#    - ScreeningResult : total, band, safety alert, and the standard "not a diagnosis" note.
#
#  🧠 DESIGN NOTES
#    - Screening, not diagnosis. Results describe symptom levels over the last two weeks and
#      always carry a note to discuss them with a GP or clinician.
#    - Safety overrides totals: any answer above 0 on PHQ-9 item 9 (thoughts of self-harm)
#      raises a safety alert, even when the total is in the "minimal" band.
#    - Only scoring is implemented. Question wording is not reproduced here; use the official
#      published forms. The PHQ and GAD-7 were developed by Drs Spitzer, Williams, Kroenke and
#      colleagues with an educational grant from Pfizer; check current terms before distribution.
#    - Bands follow the published cut-offs (5 / 10 / 15 / 20 for PHQ-9; 5 / 10 / 15 for GAD-7).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: PHQ-9 and GAD-7 scoring with safety alerts.
# ==================================================================================================
"""Scoring for validated screening questionnaires (PHQ-9, GAD-7). Not diagnostic."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

NOT_A_DIAGNOSIS = (
    "This is a screening score, not a diagnosis. Please talk to your GP or a qualified "
    "clinician about how you are feeling, especially if these symptoms continue."
)


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True, slots=True)
class ScreeningResult:
    instrument: str
    total: int
    maximum: int
    band: str
    safety_alert: bool
    note: str = NOT_A_DIAGNOSIS


def _validate(responses: Sequence[int], items: int, name: str) -> list[int]:
    answers = list(responses)
    if len(answers) != items:
        raise ValueError(f"{name} needs exactly {items} answers, got {len(answers)}")
    if any(not isinstance(a, int) or not 0 <= a <= 3 for a in answers):
        raise ValueError(f"{name} answers must be whole numbers from 0 to 3")
    return answers


def _band(total: int, cut_offs: Sequence[tuple[int, str]]) -> str:
    label = cut_offs[0][1]
    for threshold, name in cut_offs:
        if total >= threshold:
            label = name
    return label


# ========================================
# 📂 Instruments
# ========================================
PHQ9_BANDS = (
    (0, "minimal"),
    (5, "mild"),
    (10, "moderate"),
    (15, "moderately severe"),
    (20, "severe"),
)
GAD7_BANDS = ((0, "minimal"), (5, "mild"), (10, "moderate"), (15, "severe"))


def score_phq9(responses: Sequence[int]) -> ScreeningResult:
    """PHQ-9: nine answers, each 0 (not at all) to 3 (nearly every day)."""
    answers = _validate(responses, 9, "PHQ-9")
    total = sum(answers)
    return ScreeningResult(
        instrument="PHQ-9",
        total=total,
        maximum=27,
        band=_band(total, PHQ9_BANDS),
        safety_alert=answers[8] > 0,  # item 9: thoughts of self-harm, regardless of total
    )


def score_gad7(responses: Sequence[int]) -> ScreeningResult:
    """GAD-7: seven answers, each 0 (not at all) to 3 (nearly every day)."""
    answers = _validate(responses, 7, "GAD-7")
    total = sum(answers)
    return ScreeningResult(
        instrument="GAD-7",
        total=total,
        maximum=21,
        band=_band(total, GAD7_BANDS),
        safety_alert=False,
    )
