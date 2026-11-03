# ==================================================================================================
# 📜 ailab-climate :: examples :: place_report
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/examples/place_report.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    How has the climate changed where you live? Defaults to Kollam, Kerala; works for any
#    latitude / longitude on Earth.
#
#  🔧 WHAT IT DOES
#    Open-Meteo daily history (ERA5) -> six annual indices -> trends per decade with 95%
#    intervals -> optional global comparison (NASA GISTEMP) -> printed report and annual CSV
#
#  USAGE
#    python frameworks/climate/examples/place_report.py                  # Kollam
#    python frameworks/climate/examples/place_report.py --name London --lat 51.51 --lon -0.13
#    python frameworks/climate/examples/place_report.py --global --csv kollam_indices.csv
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: place climate report with optional global context and CSV export.
# ==================================================================================================
"""Climate change report for a place (default: Kollam, Kerala)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from mattslib import atomic_write

from ailab_climate import (
    INDEX_UNITS,
    GistempClient,
    OpenMeteoClient,
    annual_indices,
    place_report,
)


# ========================================
# 📂 Pipeline
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Kollam, Kerala")
    parser.add_argument("--lat", type=float, default=8.89)
    parser.add_argument("--lon", type=float, default=76.61)
    parser.add_argument("--start", type=int, default=1950, help="first year (ERA5 from 1940)")
    parser.add_argument("--end", type=int, default=date.today().year - 1, help="last full year")
    parser.add_argument("--baseline", type=int, nargs=2, default=[1961, 1990])
    parser.add_argument("--global", dest="global_", action="store_true", help="add GISTEMP")
    parser.add_argument("--csv", type=Path, help="write annual indices to this CSV file")
    args = parser.parse_args()

    weather = OpenMeteoClient().daily(
        args.lat, args.lon, date(args.start, 1, 1), date(args.end, 12, 31), place=args.name
    )
    baseline = (args.baseline[0], args.baseline[1])
    world = GistempClient().global_monthly() if args.global_ else None
    print(place_report(weather, baseline=baseline, global_anomalies=world).summary())

    if args.csv:
        indices = annual_indices(weather, baseline=baseline)
        header = "year," + ",".join(INDEX_UNITS)
        rows = [
            f"{year}," + ",".join(f"{indices.values[name][i]:.2f}" for name in INDEX_UNITS)
            for i, year in enumerate(indices.years)
        ]
        atomic_write(args.csv, "\n".join([header, *rows]) + "\n")
        print(f"\nAnnual indices written to {args.csv}")


if __name__ == "__main__":
    main()
