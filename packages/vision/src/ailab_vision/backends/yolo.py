# ==================================================================================================
# 📜 ailab-vision :: backends :: yolo
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/backends/yolo.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE). Uses Ultralytics, which is AGPL-3.0 (see package README).
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Adapts Ultralytics YOLO models (people, vehicles and 80 everyday object classes out of the
#    box) to the package's Detector interface, so applications never touch Ultralytics types.
#
#  🔧 WHAT IT DOES
#    - YoloDetector : loads a YOLO model by name or path and returns ailab Detection objects.
#                     Registered as the "yolo" backend: create_detector("yolo").
#
#  🧠 DESIGN NOTES
#    - Ultralytics is imported inside __init__, not at module level. The registry can list and
#      import this module cheaply; PyTorch is loaded only when a YOLO detector is created.
#    - A missing dependency produces an install hint, not a bare ModuleNotFoundError.
#    - Any YOLO weights file works (e.g. yolo11n.pt or a newer release such as YOLO26), so
#      upgrading the model is a configuration change, not a code change.
#    - Licence: Ultralytics is AGPL-3.0. Using this backend in a distributed or networked product
#      brings AGPL obligations or requires an Ultralytics commercial licence.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: YoloDetector wrapping Ultralytics predict().
# ==================================================================================================
"""YOLO object detection backend (Ultralytics)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from typing import Any

from ailab_vision.detector import Frame, register_backend
from ailab_vision.types import BoundingBox, Detection


# ========================================
# 📂 YOLO Backend
# ========================================
@register_backend("yolo")
class YoloDetector:
    """Ultralytics YOLO model exposed through the Detector interface."""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        *,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "the 'yolo' backend needs Ultralytics: pip install 'ailab-vision[yolo]'"
            ) from exc
        self._model: Any = YOLO(model)
        self._confidence = confidence
        self._device = device  # None lets Ultralytics choose GPU when available, else CPU

    def detect(self, frame: Frame) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._confidence, device=self._device, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names  # class index -> label, e.g. 0 -> "person"
            boxes = result.boxes
            # .tolist() converts tensors to plain Python floats in one step per field.
            for xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            ):
                detections.append(Detection(BoundingBox(*xyxy), names[int(cls)], float(conf)))
        return detections
