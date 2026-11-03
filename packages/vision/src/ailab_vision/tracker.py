# ==================================================================================================
# 📜 ailab-vision :: tracker
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/tracker.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A detector sees each frame in isolation: it can say "there is a car here" but not "this is
#    the same car as before". Counting vehicles, measuring dwell time or following a person all
#    need identity over time. This module assigns stable track ids across frames.
#
#  🔧 WHAT IT DOES
#    - IoUTracker.update(detections) : matches this frame's detections to existing tracks by
#      box overlap (IoU), returns the detections with track ids, and starts or ends tracks.
#
#  🧠 DESIGN NOTES
#    - Greedy matching on IoU, highest overlap first. Optimal (Hungarian) assignment needs
#      SciPy and rarely changes the result at video frame rates, where boxes move little
#      between frames. Greedy keeps the package dependency-free; the trade-off is documented.
#    - Labels must match by default, so a person never inherits a bicycle's track id.
#    - A track survives `max_missed` frames without a match, which bridges short occlusions
#      (someone walking behind a pole) without keeping dead tracks forever.
#    - No motion model. Fast objects or low frame rates may need a Kalman filter (as in SORT);
#      that is a planned extension, and the interface already allows swapping the tracker.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: IoUTracker with greedy matching and occlusion tolerance.
# ==================================================================================================
"""Assign stable identities to detections across video frames.

Example:
    >>> from ailab_vision.types import BoundingBox, Detection
    >>> tracker = IoUTracker()
    >>> car = Detection(BoundingBox(0, 0, 10, 10), "car", 0.9)
    >>> tracker.update([car])[0].track_id
    1
    >>> moved = Detection(BoundingBox(1, 0, 11, 10), "car", 0.9)
    >>> tracker.update([moved])[0].track_id
    1
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ailab_vision.types import BoundingBox, Detection


# ========================================
# 📂 Internal Track State
# ========================================
@dataclass
class _Track:
    track_id: int
    box: BoundingBox
    label: str
    missed: int = 0  # consecutive frames without a match


# ========================================
# 📂 IoU Tracker
# ========================================
class IoUTracker:
    """Frame-to-frame tracker based on bounding-box overlap."""

    def __init__(
        self, *, min_iou: float = 0.3, max_missed: int = 5, match_labels: bool = True
    ) -> None:
        if not 0.0 < min_iou <= 1.0:
            raise ValueError("min_iou must be in (0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")
        self._min_iou = min_iou
        self._max_missed = max_missed
        self._match_labels = match_labels
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Forget all tracks, e.g. at a scene cut or when switching video source."""
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Match ``detections`` to tracks and return them with ``track_id`` set, same order."""
        # 1. Score every plausible (track, detection) pair.
        candidates = [
            (track.box.iou(det.box), track_id, index)
            for track_id, track in self._tracks.items()
            for index, det in enumerate(detections)
            if not self._match_labels or track.label == det.label
        ]
        # 2. Greedy assignment, best overlap first; each track and detection used once.
        assigned: dict[int, int] = {}  # detection index -> track id
        used_tracks: set[int] = set()
        for overlap, track_id, index in sorted(candidates, reverse=True):
            if overlap < self._min_iou:
                break  # sorted descending: nothing further can qualify
            if track_id in used_tracks or index in assigned:
                continue
            assigned[index] = track_id
            used_tracks.add(track_id)

        # 3. Refresh matched tracks; age the unmatched ones and drop the expired.
        for index, track_id in assigned.items():
            track = self._tracks[track_id]
            track.box = detections[index].box
            track.missed = 0
        for track_id in list(self._tracks):
            if track_id not in used_tracks:
                self._tracks[track_id].missed += 1
                if self._tracks[track_id].missed > self._max_missed:
                    del self._tracks[track_id]

        # 4. Unmatched detections start new tracks.
        for index, det in enumerate(detections):
            if index not in assigned:
                assigned[index] = self._next_id
                self._tracks[self._next_id] = _Track(self._next_id, det.box, det.label)
                self._next_id += 1

        return [det.with_track(assigned[i]) for i, det in enumerate(detections)]
