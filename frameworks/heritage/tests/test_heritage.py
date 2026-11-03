# ==================================================================================================
# 📜 ailab-heritage :: tests
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/tests/test_heritage.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Tests on a synthetic landscape with known contents: gentle natural hills, a buried square
#    enclosure bank (1 m high) and a natural rounded mound. The enclosure must rank first and the
#    mound must be called likely natural; status and protection rules must hold.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest
from ailab_geo import GeoGrid

from ailab_heritage import (
    REPORTING_GUIDANCE,
    Evidence,
    SiteRecord,
    degradation,
    detect_candidates,
    generalise_location,
    public_record,
)


# ========================================
# 📂 Synthetic landscape
# ========================================
def landscape() -> npt.NDArray[np.float64]:
    n = 200
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 50 + 3 * np.sin(rows / 40) + 2 * np.cos(cols / 55)  # gentle natural hills
    z[40:80, 40:80] += 1.0  # square enclosure bank, 40 m across, 3 m wide
    z[43:77, 43:77] -= 1.0
    mound = 1.5 * np.exp(-((rows - 140) ** 2 + (cols - 140) ** 2) / (2 * 6.0**2))
    return z + mound


@pytest.fixture
def dem() -> GeoGrid:
    return GeoGrid(landscape(), origin_x=500_000.0, origin_y=1_000_000.0, cell_size=1.0)


# ========================================
# 📂 Candidates
# ========================================
def test_enclosure_ranks_first(dem: GeoGrid) -> None:
    found = detect_candidates(dem)
    top = found[0]
    assert top.kind == "raised"
    assert top.score > 0.8
    assert "enclosure" in top.interpretation
    assert top.status == "unverified candidate"
    assert top.x == pytest.approx(500_000 + 60, abs=2)
    assert top.y == pytest.approx(1_000_000 - 60, abs=2)


def test_mound_is_called_natural(dem: GeoGrid) -> None:
    mound = [c for c in detect_candidates(dem) if abs(c.x - 500_140) < 5 and c.kind == "raised"]
    assert mound
    assert mound[0].score < 0.3
    assert "natural" in mound[0].interpretation


def test_flat_ground_has_no_candidates() -> None:
    flat = np.full((60, 60), 10.0)
    flat[30:32, 30:32] += 1.0  # 4 m2 bump: below the minimum area
    assert detect_candidates(GeoGrid(flat, origin_x=0.0, origin_y=0.0, cell_size=1.0)) == []


def test_interpretations_cover_linear_and_compact() -> None:
    z = np.zeros((120, 120))
    z[20:23, 10:110] = 1.0  # long straight bank
    z[70:90, 50:70] = 1.0  # solid rectangular platform
    kinds = {c.interpretation for c in detect_candidates(GeoGrid(z, 0.0, 0.0, 1.0))}
    assert any("wall" in k for k in kinds)
    assert any("platform" in k for k in kinds)


# ========================================
# 📂 Evidence and status
# ========================================
def test_status_ladder() -> None:
    record = SiteRecord("R0001")
    record.add(Evidence("remote_sensing", "LiDAR 1 m DTM, tile TQ38"))
    assert record.status == "unverified candidate"
    record.add(Evidence("historical_map", "OS 1st edition 1:2500, 1870", note="'Castle Field'"))
    assert record.status == "supported"
    assert "Next step" in record.explain()
    record.add(Evidence("field_survey", "Survey report 2027/04"))
    assert record.status == "confirmed"
    assert "Next step" not in record.explain()
    assert "'Castle Field'" in record.explain()


def test_evidence_needs_source() -> None:
    with pytest.raises(ValueError):
        Evidence("place_name", "  ")


# ========================================
# 📂 Protection
# ========================================
def test_generalised_location_hides_the_spot() -> None:
    a = generalise_location(8.8912, 76.6141)
    b = generalise_location(8.8930, 76.6150)
    assert a == b  # nearby points share one square
    assert abs(a[0] - 8.8912) < 5 / 111.32
    with pytest.raises(ValueError):
        generalise_location(0, 0, precision_km=0)


def test_public_record_is_safe_by_default(dem: GeoGrid) -> None:
    top = detect_candidates(dem)[0]
    hidden = public_record(top, 8.8912, 76.6141, publish_exact=True)  # not confirmed
    assert hidden["location"] != {"lat": 8.8912, "lon": 76.6141}
    assert hidden["location_precision"].startswith("generalised")
    shown = public_record(top, 8.8912, 76.6141, status="confirmed", publish_exact=True)
    assert shown["location"] == {"lat": 8.8912, "lon": 76.6141}
    assert set(REPORTING_GUIDANCE) == {"IN", "UK"}


# ========================================
# 📂 Landscape change
# ========================================
def test_degradation_fades_over_time() -> None:
    z = np.zeros((40, 40))
    z[18:22, 5:35] = 1.0  # 4 m wide, 1 m high bank
    futures = degradation(z, 1.0)
    assert [f.years for f in futures] == [10, 500, 1000]
    assert futures[0].visible == "visible"
    for earlier, later in itertools.pairwise(futures):
        assert later.relief_low_m <= earlier.relief_low_m
    assert all(f.relief_low_m <= f.relief_high_m for f in futures)


def test_degradation_verdicts_and_validation() -> None:
    z = np.zeros((30, 30))
    z[14:16, 14:16] = 0.3  # small low mound
    verdicts = {f.visible for f in degradation(z, 1.0, horizons=(10, 200, 3000))}
    assert {"uncertain", "not visible"} & verdicts
    with pytest.raises(ValueError):
        degradation(z, 1.0, kappa_range=(0.01, 0.001))
