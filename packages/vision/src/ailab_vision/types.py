# ==================================================================================================
# 📜 ailab-vision :: types
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/types.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Every detection model returns results in its own format: tensors, dicts, custom classes.
#    This module defines the small, model-independent types that the rest of the package
#    speaks, so detectors, trackers and drawing code never depend on a specific model library.
#
#  🔧 WHAT IT DOES
#    - BoundingBox : an axis-aligned box in pixel coordinates, with area, centre and IoU.
#    - Detection   : one detected object (box, label, confidence and an optional track id).
#
#  🧠 DESIGN NOTES
#    - Frozen dataclasses: a detection is a fact about one frame and should never be mutated.
#      Adding a track id returns a new object (`with_track`) instead of changing the old one.
#    - Validation happens at construction, so an invalid box (x2 < x1) or a confidence outside
#      0..1 fails at the backend that produced it, not three steps later in the tracker.
#    - slots=True keeps each object small; a video can produce millions of detections.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: BoundingBox and Detection.
# ==================================================================================================
"""Model-independent result types for object detection.

Example:
    >>> a = BoundingBox(0, 0, 10, 10)
    >>> b = BoundingBox(5, 5, 15, 15)
    >>> round(a.iou(b), 3)
    0.143
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


# ========================================
# 📂 Bounding Box
# ========================================
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid box: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union: 0.0 for no overlap, 1.0 for identical boxes."""
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = ix * iy
        union = self.area + other.area - intersection
        # Two zero-area boxes have no meaningful overlap; avoid dividing by zero.
        return intersection / union if union > 0 else 0.0


# ========================================
# 📂 Detection
# ========================================
@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame."""

    box: BoundingBox
    label: str
    confidence: float
    track_id: int | None = None  # set by a tracker; None means "not tracked"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def with_track(self, track_id: int) -> Detection:
        """Return a copy of this detection carrying ``track_id``."""
        return dataclasses.replace(self, track_id=track_id)
