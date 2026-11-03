# ==================================================================================================
# 📜 ailab-vision :: examples :: detect_video
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/examples/detect_video.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    End-to-end demo: detect and track people and vehicles in a video or webcam feed, save an
#    annotated video and report throughput.
#
#  🔧 WHAT IT DOES
#    read frames -> detect (any backend) -> keep chosen labels -> track -> draw -> write video
#
#  🧠 DESIGN NOTES
#    - Uses mattslib (my core utilities) for structured logging and timing, showing the two
#      repositories working together.
#    - The backend is a command-line option: swapping models needs no code change.
#
#  USAGE
#    pip install -e "packages/vision[examples]"
#    python packages/vision/examples/detect_video.py traffic.mp4 --labels person car bus truck
#    python packages/vision/examples/detect_video.py 0            # webcam
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: detection, tracking, annotated output, throughput report.
# ==================================================================================================
"""Detect and track objects in a video and save an annotated copy."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
import logging

import cv2
from mattslib import Timer, setup_logging

from ailab_vision import IoUTracker, create_detector, filter_detections
from ailab_vision.video import draw_detections, read_frames

log = logging.getLogger("detect_video")


# ========================================
# 📂 Command Line
# ========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="video file, stream URL, or camera index such as 0")
    parser.add_argument("--backend", default="yolo")
    parser.add_argument("--model", default="yolo11n.pt", help="weights file for the backend")
    parser.add_argument("--labels", nargs="*", default=["person", "car", "bus", "truck"])
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--every", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--output", default="annotated.mp4")
    return parser.parse_args()


# ========================================
# 📂 Pipeline
# ========================================
def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    source = int(args.source) if args.source.isdigit() else args.source

    detector = create_detector(args.backend, model=args.model)
    tracker = IoUTracker()
    writer = None
    frames = 0
    seen_ids: set[int] = set()

    with Timer("pipeline") as timer:
        for frame in read_frames(source, every=args.every):
            found = filter_detections(
                detector.detect(frame), labels=args.labels, min_confidence=args.min_confidence
            )
            tracked = tracker.update(found)
            seen_ids.update(d.track_id for d in tracked if d.track_id is not None)

            if writer is None:  # open lazily, once the frame size is known
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 25, (width, height))
            writer.write(draw_detections(frame, tracked))
            frames += 1

    if writer is not None:
        writer.release()
    fps = frames / timer.elapsed if timer.elapsed else 0.0
    log.info(
        "done",
        extra={
            "frames": frames,
            "fps": round(fps, 1),
            "unique_objects": len(seen_ids),
            "output": args.output,
        },
    )


if __name__ == "__main__":
    main()
