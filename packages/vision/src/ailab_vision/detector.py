# ==================================================================================================
# 📜 ailab-vision :: detector
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/detector.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.1
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Detection models change quickly: a better YOLO appears every year, and some projects need
#    a different model family altogether. Application code should not have to change when the
#    model does. This module defines one interface for every detector and a registry that
#    selects the implementation by name.
#
#  🔧 WHAT IT DOES
#    - Detector          : the interface (a Protocol) every backend implements: detect(frame).
#    - register_backend  : decorator that adds a backend to the registry under a name.
#    - create_detector   : builds a backend by name, e.g. create_detector("yolo").
#    - available_backends: lists registered names.
#    - filter_detections : keeps only the labels and confidence levels a caller cares about.
#
#  🧠 DESIGN NOTES
#    - Protocol (structural typing): a backend does not inherit from anything; it only needs a
#      matching detect() method. Third-party classes can be adapted without touching them.
#    - Built-in backends are imported lazily, on first use. Importing this package therefore
#      never pulls in heavy libraries such as PyTorch unless that backend is actually requested.
#    - Unknown backend names fail with the list of valid ones, so a typo is a one-second fix.
#    - The registry itself is mattslib.Registry, shared with the nlp package (ADR-0003).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: Detector protocol, backend registry, filter_detections.
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Registry now provided by mattslib.Registry (extracted on second use, ADR-0003).
#      - Backend options may now include one called `name` (positional-only name argument).
# ==================================================================================================
"""A single detector interface with pluggable, lazily loaded backends."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from mattslib import Registry

from ailab_vision.types import Detection

# ========================================
# 📂 Types
# ========================================
# An image as OpenCV and most model libraries expect it: height x width x 3, BGR, uint8.
Frame = npt.NDArray[np.uint8]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns one frame into a list of detections."""

    def detect(self, frame: Frame) -> list[Detection]: ...


F = TypeVar("F", bound=Callable[..., Detector])

# ========================================
# 📂 Backend Registry
# ========================================
# Built-in backends are listed by module and imported only when requested (see DESIGN NOTES).
BACKENDS: Registry[Detector] = Registry("backend", builtins={"yolo": "ailab_vision.backends.yolo"})


def register_backend(name: str) -> Callable[[F], F]:
    """Register a detector class or factory under ``name``."""
    return BACKENDS.register(name)


def available_backends() -> list[str]:
    """Names that create_detector() accepts, including built-ins not yet loaded."""
    return BACKENDS.available()


def create_detector(name: str, /, **options: Any) -> Detector:
    """Create the backend registered as ``name``, passing ``options`` to it."""
    return BACKENDS.create(name, **options)


# ========================================
# 📂 Post-processing
# ========================================
def filter_detections(
    detections: Iterable[Detection],
    *,
    labels: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> list[Detection]:
    """Keep detections whose label is in ``labels`` (if given) and confidence >= threshold."""
    wanted = None if labels is None else set(labels)
    return [
        d
        for d in detections
        if d.confidence >= min_confidence and (wanted is None or d.label in wanted)
    ]
