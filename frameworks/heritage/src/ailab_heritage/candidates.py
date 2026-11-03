# ==================================================================================================
# 📜 ailab-heritage :: candidates
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/src/ailab_heritage/candidates.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Buried walls, banks, ditches and mounds often survive as low earthworks a few tens of
#    centimetres high. In an elevation model (ideally LiDAR) they can be found by their shape:
#    straight sides, right angles, closed rectangular outlines. This module finds such shapes and
#    ranks them as CANDIDATES for a specialist to examine. It does not declare sites.
#
#  🔧 WHAT IT DOES
#    - detect_candidates : local relief -> raised and sunken features -> shape measures ->
#                          a regularity score and a plain interpretation for each feature.
#    - Candidate         : location (map x/y), size, height or depth, shape measures, score.
#
#  🧠 DESIGN NOTES
#    - Built entirely on ailab-geo (local relief, labelling, shape measures). This module only
#      adds the archaeological reasoning: which shapes deserve attention, and why.
#    - Score = how rectangular the OUTLINE is (0.7 -> 0, 1.0 -> 1). Natural landforms are mostly
#      rounded and irregular; enclosures and building platforms tend to rectangular outlines.
#    - Both raised (banks, walls, mounds) and sunken (ditches, moats, hollow ways) features are
#      searched, because a ditch often survives better than its bank.
#    - Thresholds are in real units (metres, square metres), so results do not change with the
#      DEM resolution. Results depend on data quality: 30 m satellite DEMs miss most features
#      that 1 m LiDAR shows.
#    - Known artefact: a strong bank lowers the local mean around it, so the ground beside it can
#      appear as a shallow sunken feature. Read sunken candidates that touch a raised one as
#      possibly part of the same structure.
#    - Every candidate is labelled as unverified; see evidence.py for how status can change.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: candidate detection with shape-based regularity scoring.
# ==================================================================================================
"""Shape-based detection of possible archaeological earthworks (candidates only)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ailab_geo import GeoGrid, label_regions, local_relief, measure

Kind = Literal["raised", "sunken"]


# ========================================
# 📂 Candidate
# ========================================
@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    kind: Kind
    x: float  # map coordinates of the centroid (same CRS as the DEM)
    y: float
    area_m2: float
    relief_m: float  # mean height above (raised) or depth below (sunken) its surroundings
    outline_rectangularity: float
    fill_ratio: float
    elongation: float
    orientation_degrees: float
    score: float  # 0..1 geometric regularity
    interpretation: str
    status: str = "unverified candidate"


def _interpret(outline: float, fill: float, elongation: float, score: float) -> str:
    if score < 0.3:
        return "irregular shape: likely natural"
    if fill < 0.7:
        return "ring-like feature with a regular outline: possible enclosure"
    if elongation > 4:
        return "long straight feature: possible wall, bank, ditch or track"
    return "compact regular feature: possible platform, mound or building footprint"


# ========================================
# 📂 Detection
# ========================================
def detect_candidates(
    dem: GeoGrid,
    *,
    window_m: float = 30.0,
    min_relief_m: float = 0.25,
    min_area_m2: float = 20.0,
    max_area_m2: float = 50_000.0,
) -> list[Candidate]:
    """Raised and sunken features with their regularity score, highest score first."""
    radius = max(1, round(window_m / (2 * dem.cell_size)))
    relief = local_relief(dem.values, radius)
    cell_area = dem.cell_size**2
    candidates: list[Candidate] = []
    for kind, mask in (("raised", relief > min_relief_m), ("sunken", relief < -min_relief_m)):
        labels, _ = label_regions(np.nan_to_num(mask, nan=0.0).astype(bool))
        for region in measure(labels):
            area = region.area_cells * cell_area
            if not min_area_m2 <= area <= max_area_m2:
                continue
            cells = labels == region.label
            height = float(np.mean(np.abs(relief[cells])))
            score = float(np.clip((region.outline_rectangularity - 0.7) / 0.3, 0.0, 1.0))
            x, y = dem.to_xy(region.centroid_row, region.centroid_col)
            candidates.append(
                Candidate(
                    id=f"{kind[0].upper()}{region.label:04d}",
                    kind=kind,  # type: ignore[arg-type]
                    x=x,
                    y=y,
                    area_m2=area,
                    relief_m=height,
                    outline_rectangularity=region.outline_rectangularity,
                    fill_ratio=region.fill_ratio,
                    elongation=region.elongation,
                    orientation_degrees=region.orientation_degrees,
                    score=score,
                    interpretation=_interpret(
                        region.outline_rectangularity, region.fill_ratio, region.elongation, score
                    ),
                )
            )
    return sorted(candidates, key=lambda c: c.score, reverse=True)
