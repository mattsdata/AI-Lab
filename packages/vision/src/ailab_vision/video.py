# ==================================================================================================
# 📜 ailab-vision :: video
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/video.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The input and output side of a vision pipeline: reading frames from a file, stream or
#    webcam, and drawing results on frames so a person can check what the model saw.
#
#  🔧 WHAT IT DOES
#    - read_frames     : yields frames from a video file, stream URL or camera index.
#    - draw_detections : draws boxes, labels, confidence and track ids on a copy of a frame.
#
#  🧠 DESIGN NOTES
#    - read_frames is a generator: a two-hour video is processed one frame at a time, never
#      loaded into memory. The capture is released in `finally`, even if the caller stops early.
#    - `every` skips frames cheaply. Running a detector on every 3rd frame often gives the same
#      business answer at a third of the compute cost.
#    - draw_detections never modifies the input frame. Callers may still need the clean image
#      (for example to save it or pass it to a second model).
#    - OpenCV is optional and imported on use, with a clear install hint if missing.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: read_frames and draw_detections.
# ==================================================================================================
"""Video input and result visualisation (requires OpenCV)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from ailab_vision.detector import Frame
from ailab_vision.types import Detection

BGR = tuple[int, int, int]


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("video helpers need OpenCV: pip install 'ailab-vision[video]'") from exc
    return cv2


# ========================================
# 📂 Reading Frames
# ========================================
def read_frames(source: str | int, *, every: int = 1) -> Iterator[Frame]:
    """Yield every ``every``-th frame from a file path, stream URL or camera index (0 = webcam)."""
    if every < 1:
        raise ValueError("every must be >= 1")
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"cannot open video source: {source!r}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break  # end of file or stream closed
            if index % every == 0:
                yield frame
            index += 1
    finally:
        capture.release()


# ========================================
# 📂 Drawing Results
# ========================================
def draw_detections(
    frame: Frame,
    detections: Iterable[Detection],
    *,
    color: BGR = (0, 200, 0),
    thickness: int = 2,
) -> Frame:
    """Return a copy of ``frame`` with each detection drawn as a labelled box."""
    cv2 = _require_cv2()
    canvas: Frame = frame.copy()
    for det in detections:
        top_left = (int(det.box.x1), int(det.box.y1))
        bottom_right = (int(det.box.x2), int(det.box.y2))
        cv2.rectangle(canvas, top_left, bottom_right, color, thickness)
        text = f"{det.label} {det.confidence:.2f}"
        if det.track_id is not None:
            text += f" #{det.track_id}"
        # Keep the label inside the image when a box touches the top edge.
        origin = (top_left[0], max(top_left[1] - 5, 12))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas
