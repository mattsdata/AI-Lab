# ==================================================================================================
# 📜 ailab-mental-health :: summary
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/src/ailab_mental_health/summary.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Help a person notice patterns in their own check-ins (a dip in mood, a link with sleep)
#    in gentle, non-clinical language, and point them to support when a pattern persists or
#    when anything they write suggests they may be at risk.
#
#  🔧 WHAT IT DOES
#    - wellbeing_summary : last 7 days vs the 28 days before, sleep comparison, a sustained low
#                          mood nudge, and safety resources when a note contains crisis
#                          language. Returns a WellbeingSummary with readable text.
#
#  🧠 DESIGN NOTES
#    - Describes, never labels: "your average mood was lower than in recent weeks", never a
#      condition name. A test scans the output for diagnostic terms.
#    - Safety first in the output order: if any recent note contains crisis language, the
#      safety message comes before anything else.
#    - "Sustained low mood" (14 consecutive days with mood <= 3) leads to a suggestion to talk
#      to a GP, as a nudge, not a verdict. Thresholds are parameters for clinical review.
#    - Needs no network and no model: everything runs locally on the person's own entries.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: weekly comparison, sleep pattern, sustained-low nudge, safety first.
# ==================================================================================================
"""Gentle, non-diagnostic summary of a person's own check-ins."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ailab_mental_health.journal import CheckIn
from ailab_mental_health.safety import mentions_crisis, safety_message


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True)
class WellbeingSummary:
    as_of: date
    recent_mood: float | None  # mean of the last 7 days
    earlier_mood: float | None  # mean of the 28 days before that
    low_streak_days: int  # current run of days with mood <= low_mood
    crisis_language: bool
    text: str


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# ========================================
# 📂 Summary
# ========================================
def wellbeing_summary(
    entries: Sequence[CheckIn],
    *,
    as_of: date,
    region: str = "UK",
    low_mood: int = 3,
    sustained_days: int = 14,
) -> WellbeingSummary:
    """Compare the last week with the four weeks before it, in plain language."""
    known = sorted((e for e in entries if e.day <= as_of), key=lambda e: e.day)
    recent = [e for e in known if e.day > as_of - timedelta(days=7)]
    earlier = [e for e in known if as_of - timedelta(days=35) < e.day <= as_of - timedelta(days=7)]
    recent_mood = _mean([e.mood for e in recent])
    earlier_mood = _mean([e.mood for e in earlier])

    streak = 0
    expected = as_of
    for entry in reversed(known):  # consecutive days ending today with low mood
        if entry.day != expected or entry.mood > low_mood:
            break
        streak += 1
        expected -= timedelta(days=1)

    crisis = any(mentions_crisis(e.note) for e in recent)
    lines: list[str] = []
    if crisis:
        lines += [safety_message(region), ""]
    if recent_mood is None:
        lines.append(
            "No check-ins in the last 7 days. A short daily check-in helps you spot patterns."
        )
    else:
        lines.append(f"Your average mood over the last 7 days was {recent_mood:.1f} out of 10.")
        if earlier_mood is not None:
            difference = recent_mood - earlier_mood
            if abs(difference) < 0.5:
                lines.append(f"That is about the same as the 4 weeks before ({earlier_mood:.1f}).")
            else:
                direction = "higher" if difference > 0 else "lower"
                lines.append(f"That is {direction} than the 4 weeks before ({earlier_mood:.1f}).")
        slept = [e.sleep_hours for e in recent if e.sleep_hours is not None]
        if slept:
            lines.append(f"You slept {np.mean(slept):.1f} hours a night on average.")
    if streak >= sustained_days:
        lines.append(
            f"Your mood has been low for {streak} days in a row. It may help to talk to your GP "
            "or someone you trust about how you have been feeling."
        )
    lines.append("This summary describes your own entries. It is not a medical assessment.")
    return WellbeingSummary(as_of, recent_mood, earlier_mood, streak, crisis, "\n".join(lines))
