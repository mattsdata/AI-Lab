# ==================================================================================================
# 📜 ailab-mental-health :: safety
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/src/ailab_mental_health/safety.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Any tool that touches mental health must know what to do when someone may be at risk. This
#    module holds the crisis resources and the checks that decide when to show them, so every
#    other part of the framework uses the same, reviewed safety behaviour.
#
#  🔧 WHAT IT DOES
#    - CrisisResource, CRISIS_RESOURCES : help lines by region (UK default; IN, US included).
#    - safety_message                   : the text shown when risk is detected.
#    - mentions_crisis                  : conservative check of free text for crisis language.
#
#  🧠 DESIGN NOTES
#    - Errs towards showing help: a false alarm costs a moment, a missed one can cost much more.
#      The phrase check is a safety net, never a risk assessment or a diagnosis.
#    - Resources are data, not code paths: regions can be added or corrected without touching
#      logic. Numbers must be re-verified before any release (they change).
#    - Free text is never logged or returned by these checks; only a yes/no leaves the function.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: crisis resources (UK, IN, US), safety_message, mentions_crisis.
# ==================================================================================================
"""Crisis resources and safety checks shared by the whole framework."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import re
from dataclasses import dataclass


# ========================================
# 📂 Crisis Resources
# ========================================
@dataclass(frozen=True, slots=True)
class CrisisResource:
    name: str
    contact: str
    note: str


# Verify before each release. Emergency numbers first.
CRISIS_RESOURCES: dict[str, tuple[CrisisResource, ...]] = {
    "UK": (
        CrisisResource("Emergency services", "999", "if you or someone else is in danger now"),
        CrisisResource("Samaritans", "116 123", "free, 24 hours a day"),
        CrisisResource("NHS 111", "111", "urgent mental health help in England"),
        CrisisResource("Shout", "text SHOUT to 85258", "free text support, 24 hours a day"),
    ),
    "IN": (
        CrisisResource("Emergency services", "112", "if you or someone else is in danger now"),
        CrisisResource("Tele-MANAS", "14416", "national mental health helpline"),
    ),
    "US": (
        CrisisResource("Emergency services", "911", "if you or someone else is in danger now"),
        CrisisResource("988 Suicide & Crisis Lifeline", "call or text 988", "24 hours a day"),
    ),
}


def safety_message(region: str = "UK") -> str:
    """Supportive message with crisis contacts for ``region``."""
    try:
        resources = CRISIS_RESOURCES[region.upper()]
    except KeyError:
        raise ValueError(f"no crisis resources for {region!r}; add them before use") from None
    lines = [
        "You don't have to go through this alone. If you are thinking about harming yourself,",
        "please reach out now:",
    ]
    lines += [f"  - {r.name}: {r.contact} ({r.note})" for r in resources]
    lines.append("If you can, tell someone you trust how you are feeling.")
    return "\n".join(lines)


# ========================================
# 📂 Crisis Language Check
# ========================================
_CRISIS_PATTERNS = re.compile(
    r"\b(suicid\w*|kill(ing)? myself|end(ing)? (it all|my life)|self[- ]?harm\w*|"
    r"hurt(ing)? myself|don'?t want to (be alive|live)|better off dead|no reason to live)\b",
    re.IGNORECASE,
)


def mentions_crisis(text: str) -> bool:
    """True if the text contains crisis language. Conservative safety net, not an assessment."""
    return bool(_CRISIS_PATTERNS.search(text))
