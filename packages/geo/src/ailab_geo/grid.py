# ==================================================================================================
# 📜 ailab-geo :: grid
# --------------------------------------------------------------------------------------------------
#  File         : packages/geo/src/ailab_geo/grid.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A raster is just an array until it knows where it is on Earth. GeoGrid ties an elevation
#    model (or any raster) to map coordinates, so results can be reported as places, not pixels.
#
#  🔧 WHAT IT DOES
#    - GeoGrid       : values + top-left origin + square cell size + CRS label; pixel <-> map.
#    - read_geotiff  : loads a single-band GeoTIFF into a GeoGrid (optional `rasterio`).
#    - haversine_km  : great-circle distance between two latitude / longitude points.
#
#  🧠 DESIGN NOTES
#    - Square, north-up cells only. This covers most public DEMs (e.g. Copernicus GLO-30 after
#      reprojection, LiDAR DTMs) and keeps every algorithm simple and testable.
#    - Pixel centres: row/col (0, 0) is the centre of the top-left cell, half a cell inside
#      the origin. Off-by-half-cell errors are a classic GIS bug.
#    - rasterio (GDAL) is optional and imported only by read_geotiff.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: GeoGrid, read_geotiff, haversine_km.
# ==================================================================================================
"""Georeferenced rasters and distances.

Example:
    >>> import numpy as np
    >>> g = GeoGrid(np.zeros((2, 2)), origin_x=100.0, origin_y=50.0, cell_size=10.0)
    >>> g.to_xy(0, 0)
    (105.0, 45.0)
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


# ========================================
# 📂 GeoGrid
# ========================================
@dataclass(frozen=True)
class GeoGrid:
    values: FloatArray  # rows run north to south, columns west to east
    origin_x: float  # map x of the top-left corner
    origin_y: float  # map y of the top-left corner
    cell_size: float  # metres (or map units) per cell, square cells
    crs: str = "unknown"

    def __post_init__(self) -> None:
        if np.asarray(self.values).ndim != 2:
            raise ValueError("values must be a 2-D array")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64))

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.values.shape
        return int(rows), int(cols)

    def to_xy(self, row: float, col: float) -> tuple[float, float]:
        """Map coordinates of a cell centre."""
        return (
            self.origin_x + (col + 0.5) * self.cell_size,
            self.origin_y - (row + 0.5) * self.cell_size,
        )

    def to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Cell containing a map coordinate."""
        return (
            math.floor((self.origin_y - y) / self.cell_size),
            math.floor((x - self.origin_x) / self.cell_size),
        )

    def with_values(self, values: npt.ArrayLike) -> GeoGrid:
        """Same georeferencing, new values (e.g. slope computed from this DEM)."""
        return GeoGrid(
            np.asarray(values, dtype=np.float64),
            self.origin_x,
            self.origin_y,
            self.cell_size,
            self.crs,
        )


def read_geotiff(path: str | Path, *, band: int = 1) -> GeoGrid:
    """Load one band of a north-up GeoTIFF with square cells. Needs `rasterio`."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("read_geotiff needs rasterio: pip install 'ailab-geo[raster]'") from exc
    with rasterio.open(path) as src:
        transform: Any = src.transform
        if transform.b != 0 or transform.d != 0 or abs(transform.a) != abs(transform.e):
            raise ValueError(
                "only north-up rasters with square cells are supported; reproject first"
            )
        values = src.read(band).astype(np.float64)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
        return GeoGrid(values, transform.c, transform.f, abs(transform.a), str(src.crs))


# ========================================
# 📂 Distance
# ========================================
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres on a spherical Earth (error well under 1%)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
