# ==================================================================================================
# 📜 ailab-vision :: backends
# --------------------------------------------------------------------------------------------------
#  File         : packages/vision/src/ailab_vision/backends/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Home of the concrete detector backends. Each module adapts one model library to the
#    Detector interface and registers itself by name.
#
#  🧠 DESIGN NOTES
#    - Intentionally imports nothing. Backends are loaded on demand by create_detector(), so
#      installing the package never forces PyTorch or any other heavy dependency.
#    - Adding a backend: create a module here, decorate the class with @register_backend("name")
#      and add the module path to the `builtins` of BACKENDS in detector.py.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release with the YOLO backend.
# ==================================================================================================
"""Detector backends, loaded lazily by :func:`ailab_vision.create_detector`."""
