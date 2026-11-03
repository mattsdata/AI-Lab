# ==================================================================================================
# 📜 ailab-heritage :: landscape
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/src/ailab_heritage/landscape.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    How does a structure fade into the landscape? A collapsed wall or a castle bank slowly
#    spreads as soil creeps downhill. This module estimates how much of a feature's height
#    remains after 10, 500 or 1000 years, and whether it would still be visible in LiDAR. Run
#    forwards, it predicts degradation; read backwards, it explains why a 30 cm rise today can
#    be the remains of a much taller structure.
#
#  🔧 WHAT IT DOES
#    - degradation : remaining relief of a feature after each time horizon, for a low and a high
#                    soil diffusivity, with a LiDAR visibility verdict.
#    - FeatureFuture : the result for one horizon.
#
#  🧠 DESIGN NOTES
#    - Uses ailab-geo's diffusion model, the standard description of soil creep. It ignores
#      ploughing, floods, construction and stone robbing, which can remove features in decades;
#      results are a natural-processes estimate, stated as such.
#    - Always a range: diffusivity varies roughly from 0.001 to 0.01 m^2/yr with soil and
#      climate, so a single number would claim precision nobody has.
#    - The feature window is detrended (best-fit plane removed) first, so the natural slope of
#      the hillside is not mistaken for feature height.
#    - "Visible" means remaining relief above the vertical precision of typical airborne LiDAR
#      (about 0.15 m by default, a parameter).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: degradation over time horizons with uncertainty range.
# ==================================================================================================
"""How earthworks degrade over centuries (hillslope diffusion, with uncertainty)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ailab_geo import diffuse


@dataclass(frozen=True, slots=True)
class FeatureFuture:
    years: int
    relief_high_m: float  # remaining relief with the low diffusivity (slow erosion)
    relief_low_m: float  # remaining relief with the high diffusivity (fast erosion)
    visible: str  # "visible", "uncertain" or "not visible" in LiDAR


def _detrend(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove the best-fit plane, so a sloping hillside does not count as feature height."""
    rows, cols = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    design = np.column_stack([np.ones(z.size), rows.ravel(), cols.ravel()])
    coef, *_ = np.linalg.lstsq(design, z.ravel(), rcond=None)
    plane = (design @ coef).reshape(z.shape)
    return np.asarray(z - plane, dtype=np.float64)


def degradation(
    feature: npt.ArrayLike,
    cell_size: float,
    *,
    horizons: Sequence[int] = (10, 500, 1000),
    kappa_range: tuple[float, float] = (0.001, 0.01),
    lidar_precision_m: float = 0.15,
) -> list[FeatureFuture]:
    """Remaining relief (max minus surrounding base level) of a feature over time."""
    z = _detrend(np.asarray(feature, dtype=np.float64))
    low_k, high_k = kappa_range
    if not 0 < low_k < high_k:
        raise ValueError("kappa_range must be (low, high) with 0 < low < high")
    base = float(np.median(z))  # surrounding ground level

    def relief(surface: npt.NDArray[np.float64]) -> float:
        return float(np.max(surface)) - base

    results = []
    for years in horizons:
        slow = relief(diffuse(z, cell_size, kappa=low_k, years=years))
        fast = relief(diffuse(z, cell_size, kappa=high_k, years=years))
        if fast > lidar_precision_m:
            visible = "visible"
        elif slow > lidar_precision_m:
            visible = "uncertain"
        else:
            visible = "not visible"
        results.append(FeatureFuture(int(years), slow, fast, visible))
    return results
