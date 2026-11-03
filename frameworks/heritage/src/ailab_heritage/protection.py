# ==================================================================================================
# 📜 ailab-heritage :: protection
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/src/ailab_heritage/protection.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Publishing the exact location of an unexcavated site can lead to looting and illegal metal
#    detecting. This module makes the safe behaviour the default: public outputs show only a
#    generalised location, and the right authorities are named for reporting.
#
#  🔧 WHAT IT DOES
#    - generalise_location : snaps a latitude / longitude to the centre of a grid square of a
#                            chosen size (default 5 km), so the area is known but not the spot.
#    - public_record       : a shareable summary of a candidate with a generalised location;
#                            exact coordinates only when the record is confirmed AND the caller
#                            explicitly allows it.
#    - REPORTING_GUIDANCE  : who to tell (India, UK).
#
#  🧠 DESIGN NOTES
#    - Snapping to a fixed grid (rather than adding random noise) means repeated outputs cannot
#      be averaged to recover the true point.
#    - The longitude step widens with latitude so squares stay roughly the same size on the
#      ground everywhere.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: location generalisation, public records, reporting guidance.
# ==================================================================================================
"""Protect unexcavated sites: generalised public locations and reporting guidance."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from typing import Any

from ailab_heritage.candidates import Candidate

KM_PER_DEGREE = 111.32

REPORTING_GUIDANCE = {
    "IN": (
        "Report to the Archaeological Survey of India (ASI) circle office or the State Department "
        "of Archaeology (in Kerala, the Kerala State Archaeology Department). Protected monuments "
        "and their surroundings are governed by the Ancient Monuments and Archaeological Sites and "
        "Remains Act, 1958 and state law. Do not dig or remove material."
    ),
    "UK": (
        "Report to the local Historic Environment Record (HER) or the relevant national body "
        "(Historic England, Cadw, Historic Environment Scotland). Scheduled monuments are "
        "protected under the Ancient Monuments and Archaeological Areas Act 1979. Do not dig or "
        "use metal detectors without permission."
    ),
}


# ========================================
# 📂 Generalisation
# ========================================
def generalise_location(
    lat: float, lon: float, *, precision_km: float = 5.0
) -> tuple[float, float]:
    """Centre of the ~``precision_km`` grid square that contains the point."""
    if precision_km <= 0:
        raise ValueError("precision_km must be positive")
    lat_step = precision_km / KM_PER_DEGREE
    snapped_lat = (math.floor(lat / lat_step) + 0.5) * lat_step
    lon_step = precision_km / (KM_PER_DEGREE * max(math.cos(math.radians(snapped_lat)), 0.01))
    snapped_lon = (math.floor(lon / lon_step) + 0.5) * lon_step
    return round(snapped_lat, 4), round(snapped_lon, 4)


def public_record(
    candidate: Candidate,
    lat: float,
    lon: float,
    *,
    status: str = "unverified candidate",
    precision_km: float = 5.0,
    publish_exact: bool = False,
) -> dict[str, Any]:
    """Shareable summary. Exact location only for confirmed sites with explicit permission."""
    exact = publish_exact and status == "confirmed"
    shown_lat, shown_lon = (
        (lat, lon) if exact else generalise_location(lat, lon, precision_km=precision_km)
    )
    return {
        "id": candidate.id,
        "status": status,
        "interpretation": candidate.interpretation,
        "kind": candidate.kind,
        "area_m2": round(candidate.area_m2),
        "relief_m": round(candidate.relief_m, 2),
        "regularity_score": round(candidate.score, 2),
        "location": {"lat": shown_lat, "lon": shown_lon},
        "location_precision": "exact" if exact else f"generalised to ~{precision_km:g} km",
    }
