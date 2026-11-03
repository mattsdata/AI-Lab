# ==================================================================================================
# 📜 ailab-vision
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab vision package: reusable object detection and tracking that
#    any domain framework (finance, climate, heritage ...) can build on.
#
#  🔧 WHAT IT DOES
#    Re-exports the public API: types, the detector interface and registry, the tracker and
#    the video helpers. Video helpers are imported lazily so OpenCV stays optional.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: detection interface, YOLO backend, IoU tracker, video helpers.
# ==================================================================================================
"""ailab-vision: model-independent object detection and tracking."""

# ========================================
# 📂 Imports
# ========================================
from ailab_vision.detector import (
    Detector,
    Frame,
    available_backends,
    create_detector,
    filter_detections,
    register_backend,
)
from ailab_vision.tracker import IoUTracker
from ailab_vision.types import BoundingBox, Detection

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "BoundingBox",
    "Detection",
    "Detector",
    "Frame",
    "IoUTracker",
    "available_backends",
    "create_detector",
    "filter_detections",
    "register_backend",
]
