# ==================================================================================================
# 📜 ailab-vision :: tests :: types and tracker
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/tests/test_types_and_tracker.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the geometry and the tracking behaviour that every vision pipeline relies on.
#
#  🧠 TESTING APPROACH
#    Pure Python scenarios with hand-made boxes: no model, no video, no randomness. Each tracker
#    test describes a real situation (an object moving, two objects crossing, an occlusion).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for BoundingBox, Detection and IoUTracker.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import pytest

from ailab_vision import BoundingBox, Detection, IoUTracker


def det(x: float, label: str = "car", w: float = 10) -> Detection:
    return Detection(BoundingBox(x, 0, x + w, 10), label, 0.9)


# ========================================
# 📂 BoundingBox & Detection Tests
# ========================================
def test_box_geometry() -> None:
    box = BoundingBox(2, 4, 12, 24)
    assert (box.width, box.height, box.area, box.center) == (10, 20, 200, (7.0, 14.0))


def test_iou_cases() -> None:
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0
    assert BoundingBox(0, 0, 0, 0).iou(BoundingBox(0, 0, 0, 0)) == 0.0  # degenerate boxes


def test_invalid_box_and_confidence_are_rejected() -> None:
    with pytest.raises(ValueError):
        BoundingBox(10, 0, 0, 10)
    with pytest.raises(ValueError):
        Detection(BoundingBox(0, 0, 1, 1), "car", 1.5)


def test_with_track_returns_new_object() -> None:
    original = det(0)
    tracked = original.with_track(7)
    assert tracked.track_id == 7 and original.track_id is None


# ========================================
# 📂 Tracker Tests
# ========================================
def test_moving_object_keeps_its_id() -> None:
    tracker = IoUTracker()
    ids = [tracker.update([det(x)])[0].track_id for x in (0, 2, 4, 6)]
    assert ids == [1, 1, 1, 1]


def test_two_objects_get_distinct_ids_and_keep_them() -> None:
    tracker = IoUTracker()
    first = tracker.update([det(0), det(100)])
    second = tracker.update([det(102), det(1)])  # same objects, reversed order
    assert [d.track_id for d in first] == [1, 2]
    assert [d.track_id for d in second] == [2, 1]


def test_labels_must_match() -> None:
    tracker = IoUTracker()
    tracker.update([det(0, "person")])
    assert tracker.update([det(0, "bicycle")])[0].track_id == 2


def test_short_occlusion_is_bridged_long_one_is_not() -> None:
    tracker = IoUTracker(max_missed=2)
    tracker.update([det(0)])
    tracker.update([])
    tracker.update([])
    assert tracker.update([det(0)])[0].track_id == 1  # missed 2 frames: still alive
    for _ in range(3):
        tracker.update([])
    assert tracker.active_tracks == 0
    assert tracker.update([det(0)])[0].track_id == 2  # expired: new identity


def test_low_overlap_starts_new_track_and_reset_clears_state() -> None:
    tracker = IoUTracker(min_iou=0.5)
    tracker.update([det(0)])
    assert tracker.update([det(8)])[0].track_id == 2  # IoU ~0.11 is below 0.5
    tracker.reset()
    assert tracker.active_tracks == 0
    assert tracker.update([det(0)])[0].track_id == 1


@pytest.mark.parametrize("kwargs", [{"min_iou": 0}, {"min_iou": 1.5}, {"max_missed": -1}])
def test_tracker_validates_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        IoUTracker(**kwargs)  # type: ignore[arg-type]


def test_two_detections_competing_for_one_track() -> None:
    tracker = IoUTracker()
    tracker.update([det(0)])
    best, other = tracker.update([det(1), det(2)])  # both overlap track 1
    assert best.track_id == 1  # higher IoU wins the existing identity
    assert other.track_id == 2  # the other starts a new track
