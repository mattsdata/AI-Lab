# ==================================================================================================
# 📜 ailab-heritage :: evidence
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/src/ailab_heritage/evidence.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    "Was there a castle here?" is answered by evidence, not by an algorithm. This module records
#    each piece of evidence for a candidate (terrain, old maps, place names, texts, field survey,
#    excavation) and derives a status from it with an explicit, conservative rule.
#
#  🔧 WHAT IT DOES
#    - Evidence   : kind, source (citation), note.
#    - SiteRecord : a candidate with its evidence; `status` and `explain()`.
#
#  🧠 DESIGN NOTES
#    - Status ladder:
#        unverified candidate -> remote sensing only
#        supported            -> plus independent documentary evidence (texts, maps, place names)
#        confirmed            -> plus field survey or excavation by a qualified team
#    - Software can never move a record to "confirmed"; only field evidence can. Strong terrain
#      evidence, however regular, stays a candidate.
#    - Every item needs a source, so a reviewer can check it; uncited evidence is rejected.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: evidence kinds, status ladder, explanations.
# ==================================================================================================
"""Evidence records and a conservative status for possible heritage sites."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceKind = Literal[
    "remote_sensing",
    "historical_text",
    "historical_map",
    "place_name",
    "field_survey",
    "excavation",
]
DOCUMENTARY = frozenset({"historical_text", "historical_map", "place_name"})
FIELD = frozenset({"field_survey", "excavation"})


# ========================================
# 📂 Records
# ========================================
@dataclass(frozen=True, slots=True)
class Evidence:
    kind: EvidenceKind
    source: str  # citation: dataset, archive reference, survey report
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("every piece of evidence needs a source")


@dataclass
class SiteRecord:
    candidate_id: str
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, item: Evidence) -> None:
        self.evidence.append(item)

    @property
    def status(self) -> str:
        kinds = {e.kind for e in self.evidence}
        if kinds & FIELD:
            return "confirmed"
        if kinds & DOCUMENTARY:
            return "supported"
        return "unverified candidate"

    def explain(self) -> str:
        lines = [f"{self.candidate_id}: {self.status}"]
        lines += [
            f"  - {e.kind}: {e.source}" + (f" ({e.note})" if e.note else "") for e in self.evidence
        ]
        if self.status != "confirmed":
            lines.append(
                "  Next step: field survey by a qualified team, with landowner permission."
            )
        return "\n".join(lines)
