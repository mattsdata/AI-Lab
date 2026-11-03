# ==================================================================================================
# 📜 ailab-geo :: regions
# --------------------------------------------------------------------------------------------------
#  File         : packages/geo/src/ailab_geo/regions.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Turn a yes/no map (e.g. "raised above its surroundings") into separate objects and measure
#    their shape. Shape is often what separates human-made features (straight sides, right
#    angles, regular widths) from natural ones (rounded, irregular).
#
#  🔧 WHAT IT DOES
#    - label_regions : connected components of a boolean mask (4- or 8-connectivity).
#    - measure       : per region: area, centroid, orientation, elongation, rectangularity,
#                      and fill ratio (how solid versus ring-like it is).
#    - Region        : the measurements.
#
#  🧠 DESIGN NOTES
#    - Pure NumPy breadth-first labelling: no SciPy, deterministic, easy to audit.
#    - Orientation and elongation come from the covariance (principal axes) of the pixels.
#    - Rectangularity = pixel area / area of the smallest box aligned with the principal axes:
#      1.0 for a solid rectangle, pi/4 (0.785) for a large disc, less for small pixel discs.
#    - Fill ratio = pixel area / (area including enclosed holes): a ring-shaped earthwork, such
#      as an enclosure bank, has a low fill ratio; a solid mound has 1.0.
#    - Outline rectangularity uses the filled shape, so a square ring scores like a square.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: labelling and shape measurements.
# ==================================================================================================
"""Connected regions and their shape measurements."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


# ========================================
# 📂 Labelling
# ========================================
def label_regions(mask: npt.ArrayLike, *, connectivity: int = 8) -> tuple[IntArray, int]:
    """Label connected True cells 1..n (0 = background). Returns (labels, n)."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or connectivity not in (4, 8):
        raise ValueError("mask must be 2-D and connectivity 4 or 8")
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    labels = np.zeros(m.shape, dtype=np.int64)
    rows, cols = m.shape
    current = 0
    for r0, c0 in zip(*np.nonzero(m), strict=True):
        if labels[r0, c0]:
            continue
        current += 1
        labels[r0, c0] = current
        queue = deque([(int(r0), int(c0))])
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and m[nr, nc] and not labels[nr, nc]:
                    labels[nr, nc] = current
                    queue.append((nr, nc))
    return labels, current


# ========================================
# 📂 Shape Measurements
# ========================================
@dataclass(frozen=True, slots=True)
class Region:
    label: int
    area_cells: int
    centroid_row: float
    centroid_col: float
    orientation_degrees: float  # long axis, 0 = east-west, 90 = north-south
    elongation: float  # long axis / short axis, >= 1
    rectangularity: float  # area / principal-axis bounding box area, 0..1
    fill_ratio: float  # area / area with holes filled, 0..1 (low = ring-like)
    outline_rectangularity: float  # filled area / bounding box area: shape of the outline


def _filled_area(cells: npt.NDArray[np.bool_]) -> int:
    """Pixels inside the region's outer boundary (holes filled): flood the outside first."""
    padded = np.pad(cells, 1)
    outside = np.zeros(padded.shape, dtype=bool)
    outside[0, 0] = True
    queue = deque([(0, 0)])
    rows, cols = padded.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not outside[nr, nc] and not padded[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return int(padded.size - outside.sum())


def measure(labels: npt.ArrayLike) -> list[Region]:
    """Shape measurements for every labelled region, in label order."""
    lab = np.asarray(labels, dtype=np.int64)
    regions = []
    for label in range(1, (int(np.max(lab)) if lab.size else 0) + 1):
        rr, cc = np.nonzero(lab == label)
        if len(rr) == 0:
            continue
        cy, cx = float(rr.mean()), float(cc.mean())
        if len(rr) >= 2:
            cov = np.cov(np.vstack([cc - cx, -(rr - cy)]))  # x east, y north
            eigvals, eigvecs = np.linalg.eigh(cov)
            major = eigvecs[:, 1]
            angle = math.degrees(math.atan2(major[1], major[0])) % 180.0
            elongation = math.sqrt(eigvals[1] / max(eigvals[0], 1e-12)) if eigvals[1] > 0 else 1.0
            u = (cc - cx) * major[0] + (-(rr - cy)) * major[1]
            v = (cc - cx) * -major[1] + (-(rr - cy)) * major[0]
            box = (np.ptp(u) + 1) * (np.ptp(v) + 1)
        else:
            angle, elongation, box = 0.0, 1.0, 1.0
        r0, c0 = rr.min(), cc.min()
        local = np.zeros((rr.max() - r0 + 1, cc.max() - c0 + 1), dtype=bool)
        local[rr - r0, cc - c0] = True
        filled = _filled_area(local)
        regions.append(
            Region(
                label=label,
                area_cells=len(rr),
                centroid_row=cy,
                centroid_col=cx,
                orientation_degrees=angle,
                elongation=float(min(elongation, 1e6)),
                rectangularity=float(min(1.0, len(rr) / box)),
                fill_ratio=len(rr) / filled,
                outline_rectangularity=float(min(1.0, filled / box)),
            )
        )
    return regions
