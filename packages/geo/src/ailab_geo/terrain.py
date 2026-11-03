# ==================================================================================================
# 📜 ailab-geo :: terrain
# --------------------------------------------------------------------------------------------------
#  File         : packages/geo/src/ailab_geo/terrain.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Read the shape of the land from an elevation model: how steep, which way it faces, how it
#    looks under low light, which small bumps and hollows stand out from the wider landscape,
#    and how a landform softens over centuries as soil creeps downhill.
#
#  🔧 WHAT IT DOES
#    - slope_degrees, aspect_degrees : steepness and facing direction (0 = north, clockwise).
#    - hillshade                     : illumination from a chosen sun azimuth and altitude.
#    - local_relief                  : elevation minus its neighbourhood mean (a Local Relief
#                                      Model); removes hills and valleys, keeps small features.
#    - diffuse                       : hillslope erosion as linear diffusion,
#                                      dz/dt = k * Laplacian(z).
#
#  🧠 DESIGN NOTES
#    - Gradients use central differences (numpy.gradient) in map units, so slopes are correct
#      for any cell size, not just 1 m.
#    - local_relief uses an integral image (summed-area table): O(1) per cell for any radius.
#      Edges use the cells that exist, never padding with invented elevations.
#    - Diffusion is the textbook model of soil creep on soil-mantled slopes. The time step is
#      chosen automatically below the stability limit dt <= cell^2 / (4k); boundaries are
#      closed so volume is conserved. Typical k is about 0.001-0.01 m^2/yr; results should be
#      shown for a range of k, because the right value depends on soil and climate.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: slope, aspect, hillshade, local relief, diffusion.
# ==================================================================================================
"""Terrain analysis and landform change on elevation models."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _gradients(dem: npt.ArrayLike, cell_size: float) -> tuple[FloatArray, FloatArray]:
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("dem must be a 2-D array with at least 2 x 2 cells")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    dz_drow, dz_dcol = np.gradient(z, cell_size)
    # Rows increase southwards, so the northward gradient is the negative row gradient.
    east: FloatArray = np.asarray(dz_dcol, dtype=np.float64)
    north: FloatArray = -np.asarray(dz_drow, dtype=np.float64)
    return east, north


# ========================================
# 📂 Slope, Aspect, Hillshade
# ========================================
def slope_degrees(dem: npt.ArrayLike, cell_size: float) -> FloatArray:
    east, north = _gradients(dem, cell_size)
    out: FloatArray = np.degrees(np.arctan(np.hypot(east, north)))
    return out


def aspect_degrees(dem: npt.ArrayLike, cell_size: float) -> FloatArray:
    """Direction the slope faces (downhill), 0 = north, 90 = east; NaN on flat ground."""
    east, north = _gradients(dem, cell_size)
    aspect = (np.degrees(np.arctan2(-east, -north)) + 360.0) % 360.0
    out: FloatArray = np.where(np.hypot(east, north) < 1e-12, np.nan, aspect)
    return out


def hillshade(
    dem: npt.ArrayLike, cell_size: float, *, azimuth: float = 315.0, altitude: float = 45.0
) -> FloatArray:
    """Brightness 0..1 for a sun at ``azimuth`` (from north, clockwise) and ``altitude``."""
    east, north = _gradients(dem, cell_size)
    zenith = math.radians(90.0 - altitude)
    sun = math.radians(azimuth)
    slope = np.arctan(np.hypot(east, north))
    aspect = np.arctan2(-east, -north)
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(sun - aspect)
    out: FloatArray = np.clip(shade, 0.0, 1.0)
    return out


# ========================================
# 📂 Local Relief Model
# ========================================
def local_relief(dem: npt.ArrayLike, radius: int) -> FloatArray:
    """Elevation minus the mean of the (2r+1) x (2r+1) neighbourhood; NaN cells are ignored."""
    z = np.asarray(dem, dtype=np.float64)
    if z.ndim != 2 or radius < 1:
        raise ValueError("dem must be 2-D and radius >= 1")
    valid = ~np.isnan(z)
    filled = np.where(valid, z, 0.0)

    def box_sum(a: FloatArray) -> FloatArray:
        table = np.pad(a, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        rows, cols = a.shape
        r0 = np.clip(np.arange(rows) - radius, 0, rows)
        r1 = np.clip(np.arange(rows) + radius + 1, 0, rows)
        c0 = np.clip(np.arange(cols) - radius, 0, cols)
        c1 = np.clip(np.arange(cols) + radius + 1, 0, cols)
        out: FloatArray = table[r1][:, c1] - table[r0][:, c1] - table[r1][:, c0] + table[r0][:, c0]
        return out

    counts = box_sum(valid.astype(np.float64))
    mean = box_sum(filled) / np.maximum(counts, 1.0)
    out: FloatArray = np.where(valid, z - mean, np.nan)
    return out


# ========================================
# 📂 Landform Change
# ========================================
def diffuse(dem: npt.ArrayLike, cell_size: float, *, kappa: float, years: float) -> FloatArray:
    """Hillslope diffusion for ``years`` with diffusivity ``kappa`` (m^2/yr); volume conserved."""
    z: FloatArray = np.array(dem, dtype=np.float64, copy=True)
    if z.ndim != 2 or np.isnan(z).any():
        raise ValueError("dem must be a 2-D array without NaN")
    if kappa <= 0 or years < 0 or cell_size <= 0:
        raise ValueError("kappa and cell_size must be positive and years >= 0")
    if years == 0:
        return z
    limit = cell_size**2 / (4 * kappa)  # explicit-scheme stability limit
    steps = max(1, math.ceil(years / (0.9 * limit)))
    dt = years / steps
    factor = kappa * dt / cell_size**2
    for _ in range(steps):
        padded = np.pad(z, 1, mode="edge")  # closed boundary: no flux across the edge
        laplacian = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * z
        )
        z = np.asarray(z + factor * laplacian, dtype=np.float64)
    return z
