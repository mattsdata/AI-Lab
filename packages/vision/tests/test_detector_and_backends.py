# ==================================================================================================
# 📜 ailab-vision :: tests :: detector and backends
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/tests/test_detector_and_backends.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the detector registry, filtering, and the YOLO adapter's conversion logic.
#
#  🧠 TESTING APPROACH
#    The YOLO backend is tested against a fake `ultralytics` module that returns results in
#    Ultralytics' shape. This checks our adapter (the code we own) in milliseconds, with no
#    model download, GPU or PyTorch. The real model is exercised by the example script.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for the registry, filter_detections and YoloDetector.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from ailab_vision import (
    BoundingBox,
    Detection,
    Detector,
    available_backends,
    create_detector,
    filter_detections,
    register_backend,
)
from ailab_vision import detector as detector_module

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


# ========================================
# 📂 Registry Tests
# ========================================
@pytest.fixture
def clean_registry() -> Iterator[None]:
    before = set(detector_module.BACKENDS.available())
    yield
    for name in set(detector_module.BACKENDS.available()) - before:
        detector_module.BACKENDS.unregister(name)


def test_custom_backend_round_trip(clean_registry: None) -> None:
    @register_backend("fixed")
    class Fixed:
        def __init__(self, label: str = "person") -> None:
            self.label = label

        def detect(self, frame: Any) -> list[Detection]:
            return [Detection(BoundingBox(0, 0, 1, 1), self.label, 0.8)]

    backend = create_detector("fixed", label="car")
    assert isinstance(backend, Detector)
    assert backend.detect(FRAME)[0].label == "car"
    assert "fixed" in available_backends() and "yolo" in available_backends()


def test_duplicate_and_unknown_backends_fail_clearly(clean_registry: None) -> None:
    register_backend("dup")(lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("dup")(lambda: None)
    with pytest.raises(ValueError, match=r"available: .*yolo"):
        create_detector("does-not-exist")


def test_filter_detections() -> None:
    dets = [
        Detection(BoundingBox(0, 0, 1, 1), "person", 0.9),
        Detection(BoundingBox(0, 0, 1, 1), "car", 0.4),
        Detection(BoundingBox(0, 0, 1, 1), "dog", 0.95),
    ]
    assert [d.label for d in filter_detections(dets, labels=["person", "car"])] == [
        "person",
        "car",
    ]
    assert [d.label for d in filter_detections(dets, min_confidence=0.5)] == ["person", "dog"]


# ========================================
# 📂 YOLO Backend Tests (fake Ultralytics)
# ========================================
class _FakeYOLO:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def predict(self, frame: Any, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        boxes = types.SimpleNamespace(
            xyxy=np.array([[1.0, 2.0, 11.0, 12.0], [5.0, 5.0, 9.0, 9.0]]),
            conf=np.array([0.91, 0.42]),
            cls=np.array([0.0, 2.0]),
        )
        return [types.SimpleNamespace(names={0: "person", 2: "car"}, boxes=boxes)]


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch, clean_registry: None) -> None:
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    # Force a fresh import so the backend registers against the clean registry.
    monkeypatch.delitem(sys.modules, "ailab_vision.backends.yolo", raising=False)
    detector_module.BACKENDS.unregister("yolo")


def test_yolo_backend_converts_results(fake_ultralytics: None) -> None:
    yolo = create_detector("yolo", model="custom.pt", confidence=0.3, device="cpu")
    found = yolo.detect(FRAME)
    assert [(d.label, round(d.confidence, 2)) for d in found] == [("person", 0.91), ("car", 0.42)]
    assert found[0].box == BoundingBox(1.0, 2.0, 11.0, 12.0)
    assert yolo._model.model == "custom.pt"  # type: ignore[attr-defined]
    assert yolo._model.calls == [{"conf": 0.3, "device": "cpu", "verbose": False}]  # type: ignore[attr-defined]


def test_yolo_backend_explains_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, clean_registry: None
) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # makes the import fail
    monkeypatch.delitem(sys.modules, "ailab_vision.backends.yolo", raising=False)
    detector_module.BACKENDS.unregister("yolo")
    with pytest.raises(ImportError, match=r"ailab-vision\[yolo\]"):
        create_detector("yolo")
