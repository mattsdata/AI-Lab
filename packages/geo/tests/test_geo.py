# ==================================================================================================
# 📜 ailab-geo :: tests
# --------------------------------------------------------------------------------------------------
#  File         : packages/geo/tests/test_geo.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves every function against geometry with a known answer: tilted planes for slope and
#    aspect, flat ground for hillshade, a real city-pair distance, conservation of volume for
#    erosion, and simple shapes (rectangle, disc, ring) for region measurements.
#
#  🧠 TESTING APPROACH
#    Analytic surfaces and hand-drawn masks, plus a fake `rasterio` module for GeoTIFF reading.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-geo.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from ailab_geo import (
    GeoGrid,
    aspect_degrees,
    diffuse,
    haversine_km,
    hillshade,
    label_regions,
    local_relief,
    measure,
    read_geotiff,
    slope_degrees,
)

ROWS, COLS = 40, 50
_rows, _cols = np.mgrid[0:ROWS, 0:COLS].astype(float)


# ========================================
# 📂 Grid and Distance Tests
# ========================================
def test_grid_coordinates_round_trip() -> None:
    grid = GeoGrid(
        np.zeros((3, 4)), origin_x=1000.0, origin_y=5000.0, cell_size=2.0, crs="EPSG:27700"
    )
    assert grid.shape == (3, 4)
    x, y = grid.to_xy(2, 3)
    assert (x, y) == (1007.0, 4995.0) and grid.to_rowcol(x, y) == (2, 3)
    assert grid.with_values(np.ones((3, 4))).crs == "EPSG:27700"
    with pytest.raises(ValueError):
        GeoGrid(np.zeros(3), 0, 0, 1)
    with pytest.raises(ValueError):
        GeoGrid(np.zeros((2, 2)), 0, 0, 0)


def test_haversine_matches_known_distances() -> None:
    assert haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.5, abs=2)
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.05)


def test_read_geotiff_with_fake_rasterio(monkeypatch: pytest.MonkeyPatch) -> None:
    class Source:
        transform = types.SimpleNamespace(a=5.0, b=0.0, c=100.0, d=0.0, e=-5.0, f=900.0)
        nodata = -9999.0
        crs = "EPSG:32643"

        def __enter__(self) -> "Source":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, band: int) -> Any:
            return np.array([[1.0, -9999.0], [3.0, 4.0]])

    module = types.ModuleType("rasterio")
    module.open = lambda path: Source()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rasterio", module)
    grid = read_geotiff("dem.tif")
    assert grid.cell_size == 5.0 and grid.origin_y == 900.0 and grid.crs == "EPSG:32643"
    assert np.isnan(grid.values[0, 1])  # nodata becomes NaN
    Source.transform = types.SimpleNamespace(a=5.0, b=1.0, c=0.0, d=0.0, e=-5.0, f=0.0)
    with pytest.raises(ValueError, match="north-up"):
        read_geotiff("rotated.tif")
    monkeypatch.setitem(sys.modules, "rasterio", None)
    with pytest.raises(ImportError, match=r"ailab-geo\[raster\]"):
        read_geotiff("dem.tif")


# ========================================
# 📂 Terrain Tests
# ========================================
def test_slope_and_aspect_on_tilted_planes() -> None:
    rises_east = 0.1 * _cols * 2.0  # 10% gradient with 2 m cells
    assert slope_degrees(rises_east, 2.0) == pytest.approx(np.full((ROWS, COLS), 5.7106), abs=1e-3)
    assert np.allclose(aspect_degrees(rises_east, 2.0), 270.0)  # downhill faces west
    rises_north = -0.1 * _rows  # rows increase southwards
    assert np.allclose(aspect_degrees(rises_north, 1.0), 180.0)  # faces south
    assert np.isnan(aspect_degrees(np.zeros((5, 5)), 1.0)).all()  # flat: no aspect
    with pytest.raises(ValueError):
        slope_degrees(np.zeros(5), 1.0)
    with pytest.raises(ValueError):
        slope_degrees(np.zeros((5, 5)), 0.0)


def test_hillshade_lights_slopes_facing_the_sun() -> None:
    assert hillshade(np.zeros((5, 5)), 1.0, altitude=45.0) == pytest.approx(
        np.full((5, 5), math.sqrt(0.5))
    )
    faces_west = 0.5 * _cols  # sun from the north-west by default
    faces_east = -0.5 * _cols
    assert hillshade(faces_west, 1.0).mean() > hillshade(faces_east, 1.0).mean()


def test_local_relief_keeps_small_features() -> None:
    hill = 50 * np.exp(-((_rows - 20) ** 2 + (_cols - 25) ** 2) / 400)  # broad natural hill
    mound = hill.copy()
    mound[18:22, 23:27] += 1.0  # small 1 m feature on top
    relief = local_relief(mound, radius=5)
    assert relief[20, 25] > 0.5  # the feature stands out
    assert abs(local_relief(np.full((10, 10), 7.0), radius=3)).max() < 1e-12
    with_gap = mound.copy()
    with_gap[0, 0] = np.nan
    assert np.isnan(local_relief(with_gap, 2)[0, 0]) and np.isfinite(
        local_relief(with_gap, 2)[1, 1]
    )
    with pytest.raises(ValueError):
        local_relief(mound, 0)


def test_diffusion_softens_landforms_and_conserves_volume() -> None:
    bank = np.zeros((30, 30))
    bank[10:20, 14:16] = 2.0  # a 2 m high, 2 m wide earthwork bank (1 m cells)
    after_500 = diffuse(bank, 1.0, kappa=0.005, years=500)
    after_1000 = diffuse(bank, 1.0, kappa=0.005, years=1000)
    assert after_1000.sum() == pytest.approx(bank.sum())  # soil moves, it is not lost
    assert bank.max() > after_500.max() > after_1000.max()  # the bank lowers over time
    faster = diffuse(bank, 1.0, kappa=0.01, years=500)
    assert faster.max() < after_500.max()
    assert np.array_equal(diffuse(bank, 1.0, kappa=0.005, years=0), bank)
    huge = diffuse(bank, 1.0, kappa=50.0, years=10)  # stability handled by sub-stepping
    assert np.isfinite(huge).all() and huge.max() <= bank.max()
    with pytest.raises(ValueError):
        diffuse(bank, 1.0, kappa=0.0, years=10)
    with pytest.raises(ValueError):
        diffuse(np.full((3, 3), np.nan), 1.0, kappa=0.01, years=10)


# ========================================
# 📂 Region Tests
# ========================================
def test_labelling_and_connectivity() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = mask[1, 1] = True  # diagonal neighbours
    mask[4:6, 4:6] = True
    assert label_regions(mask, connectivity=8)[1] == 2
    assert label_regions(mask, connectivity=4)[1] == 3
    assert label_regions(np.zeros((3, 3), dtype=bool))[1] == 0
    with pytest.raises(ValueError):
        label_regions(mask, connectivity=6)


def test_shapes_are_measured_correctly() -> None:
    r, c = np.mgrid[0:60, 0:80]
    mask = np.zeros((60, 80), dtype=bool)
    mask[5:9, 5:35] = True  # solid 4 x 30 east-west rectangle
    mask |= (r - 40) ** 2 + (c - 20) ** 2 <= 64  # disc of radius 8
    ring = np.zeros_like(mask)
    ring[30:50, 50:70] = True
    ring[33:47, 53:67] = False  # square enclosure bank, 3 cells wide
    mask |= ring
    labels, n = label_regions(mask)
    regions = measure(labels)
    rectangle = next(x for x in regions if x.area_cells == 120)
    circle = next(x for x in regions if abs(x.centroid_row - 40) < 1 and x.centroid_col < 40)
    enclosure = next(x for x in regions if abs(x.centroid_col - 59.5) < 1)
    assert n == 3
    assert (
        rectangle.rectangularity == pytest.approx(1.0) and rectangle.orientation_degrees % 180 < 1
    )
    assert rectangle.elongation > 5 and rectangle.fill_ratio == 1.0
    # A pixel disc of radius 8 fills 197 of its 17 x 17 box (pi/4 only in the continuous limit).
    assert circle.rectangularity == pytest.approx(197 / 17**2) and circle.elongation < 1.1
    assert enclosure.fill_ratio == pytest.approx(204 / 400)  # ring-like: bank, empty interior
    assert enclosure.outline_rectangularity == pytest.approx(1.0)  # but its outline is square
    single = np.zeros((3, 3), dtype=int)
    single[1, 1] = 1
    (dot,) = measure(single)
    assert (dot.area_cells, dot.elongation, dot.rectangularity) == (1, 1.0, 1.0)

    gappy = np.zeros((3, 3), dtype=int)
    gappy[0, 0], gappy[2, 2] = 1, 3  # label 2 unused
    assert [x.label for x in measure(gappy)] == [1, 3]
