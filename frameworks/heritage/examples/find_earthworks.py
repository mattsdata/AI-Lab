# ==================================================================================================
# 📜 ailab-heritage :: examples/find_earthworks.py
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/heritage/examples/find_earthworks.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    End-to-end demo: elevation model -> ranked candidates -> evidence status -> safe public
#    record -> how the top feature would fade over 10, 500 and 1000 years.
#
#  🔧 HOW TO RUN
#    python examples/find_earthworks.py                    # synthetic landscape
#    python examples/find_earthworks.py dtm.tif 8.89 76.61 # your GeoTIFF + its approx lat/lon
#    (a GeoTIFF needs: pip install "ailab-geo[raster] @ git+...#subdirectory=packages/geo")
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""Find candidate earthworks in an elevation model and report them safely."""

from __future__ import annotations

import json
import sys

import numpy as np
from ailab_geo import GeoGrid, read_geotiff

from ailab_heritage import (
    REPORTING_GUIDANCE,
    Evidence,
    SiteRecord,
    degradation,
    detect_candidates,
    public_record,
)


def synthetic() -> GeoGrid:
    """Gentle hills with a buried square enclosure bank and a natural mound."""
    rows, cols = np.mgrid[0:200, 0:200].astype(np.float64)
    z = 50 + 3 * np.sin(rows / 40) + 2 * np.cos(cols / 55)
    z[40:80, 40:80] += 1.0
    z[43:77, 43:77] -= 1.0
    z += 1.5 * np.exp(-((rows - 140) ** 2 + (cols - 140) ** 2) / 72.0)
    return GeoGrid(z, origin_x=0.0, origin_y=0.0, cell_size=1.0)


def main() -> None:
    if len(sys.argv) >= 4:
        dem, lat, lon = read_geotiff(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    else:
        dem, lat, lon = synthetic(), 8.89, 76.61

    candidates = detect_candidates(dem)
    print(f"{len(candidates)} candidate features (highest regularity first)\n")
    for c in candidates[:10]:
        print(
            f"  {c.id}  {c.kind:6}  score {c.score:.2f}  {c.area_m2:7.0f} m2  "
            f"{c.relief_m:.2f} m  {c.interpretation}"
        )
    if not candidates:
        return

    top = candidates[0]
    record = SiteRecord(top.id, [Evidence("remote_sensing", "elevation model analysed above")])
    print("\n" + record.explain())

    print("\nPublic record (safe to share):")
    print(json.dumps(public_record(top, lat, lon, status=record.status), indent=2))

    r0, c0 = dem.to_rowcol(top.x, top.y)
    half = int(np.sqrt(top.area_m2) / dem.cell_size) + 10
    window = dem.values[max(0, r0 - half) : r0 + half + 1, max(0, c0 - half) : c0 + half + 1]
    print("\nIf left to natural soil creep (slow .. fast):")
    for f in degradation(window, dem.cell_size):
        print(
            f"  {f.years:5d} years: {f.relief_low_m:.2f} .. {f.relief_high_m:.2f} m  ({f.visible})"
        )

    print("\nReporting:", REPORTING_GUIDANCE["IN"])


if __name__ == "__main__":
    main()
