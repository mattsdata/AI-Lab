# ==================================================================================================
# 📜 ailab-vision :: tests :: video
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/tests/test_video.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves frame reading, frame skipping, drawing and the missing-OpenCV error path.
#
#  🧠 TESTING APPROACH
#    Each test writes a tiny synthetic video into pytest's tmp_path, so no sample files are
#    stored in the repository and every run starts from a known state.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for read_frames and draw_detections.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import sys
from pathlib import Path

import numpy as np
import pytest

from ailab_vision import BoundingBox, Detection

cv2 = pytest.importorskip("cv2")
from ailab_vision.video import draw_detections, read_frames  # noqa: E402


def write_video(path: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(frames):
        writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


# ========================================
# 📂 Reading Tests
# ========================================
def test_reads_all_frames_and_skips(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.avi", frames=6)
    assert len(list(read_frames(str(video)))) == 6
    assert len(list(read_frames(str(video), every=2))) == 3


def test_bad_source_and_bad_step(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="cannot open"):
        list(read_frames(str(tmp_path / "missing.avi")))
    with pytest.raises(ValueError):
        list(read_frames("x", every=0))


# ========================================
# 📂 Drawing Tests
# ========================================
def test_draw_does_not_modify_input() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [Detection(BoundingBox(5, 5, 30, 30), "person", 0.9, track_id=3)]
    drawn = draw_detections(frame, dets)
    assert frame.sum() == 0  # original untouched
    assert drawn.sum() > 0  # copy has the box and label


def test_missing_opencv_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match=r"ailab-vision\[video\]"):
        draw_detections(np.zeros((4, 4, 3), dtype=np.uint8), [])
