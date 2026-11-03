# ==================================================================================================
# 📜 ailab-geo
# --------------------------------------------------------------------------------------------------
#  File         : packages/geo/src/ailab_geo/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab geospatial package: georeferenced rasters, terrain analysis,
#    landform change and region shape measurement, for any domain that reads the land.
#
#  🔧 WHAT IT DOES
#    Re-exports GeoGrid, terrain functions, diffusion, labelling and shape measures.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: grid, terrain, diffusion, regions, distance.
# ==================================================================================================
"""ailab-geo: terrain analysis, landform change and shape measurement (NumPy only)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_geo.grid import GeoGrid, haversine_km, read_geotiff
from ailab_geo.regions import Region, label_regions, measure
from ailab_geo.terrain import (
    aspect_degrees,
    diffuse,
    hillshade,
    local_relief,
    slope_degrees,
)

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "GeoGrid",
    "Region",
    "aspect_degrees",
    "diffuse",
    "haversine_km",
    "hillshade",
    "label_regions",
    "local_relief",
    "measure",
    "read_geotiff",
    "slope_degrees",
]
