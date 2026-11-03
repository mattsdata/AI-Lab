# ailab-vision

*Author: **Mathew Varghese** · Version 1.1*

Model-independent object detection and tracking. Application code talks to one small
interface; the model behind it (YOLO today, something else tomorrow) is a configuration choice.

```python
from ailab_vision import IoUTracker, create_detector, filter_detections
from ailab_vision.video import read_frames, draw_detections

detector = create_detector("yolo", model="yolo11n.pt")  # any YOLO weights, incl. newer releases
tracker = IoUTracker()

for frame in read_frames("traffic.mp4", every=2):
    people_and_cars = filter_detections(detector.detect(frame), labels=["person", "car"])
    for d in tracker.update(people_and_cars):
        print(d.track_id, d.label, f"{d.confidence:.2f}", d.box)
```

## Components

| Module | What it does |
|---|---|
| `types` | `BoundingBox` (area, centre, IoU) and `Detection`, independent of any model library |
| `detector` | `Detector` protocol, backend registry (`create_detector`), `filter_detections` |
| `backends/yolo` | Ultralytics YOLO adapter, registered as `"yolo"` |
| `tracker` | `IoUTracker`: stable ids across frames, label-aware, tolerates short occlusions |
| `video` | `read_frames` (file, stream, webcam) and `draw_detections` |

## Install

```bash
pip install -e "packages/vision"            # core: numpy only
pip install -e "packages/vision[yolo]"      # + YOLO backend (PyTorch)
pip install -e "packages/vision[video]"     # + OpenCV video helpers
pip install -e "packages/vision[examples]"  # everything for the demo, incl. mattslib
```

## Demo

```bash
python packages/vision/examples/detect_video.py traffic.mp4 --labels person car bus truck
python packages/vision/examples/detect_video.py 0     # webcam
```

Writes `annotated.mp4` and logs frames processed, throughput (fps) and unique objects tracked.

## Adding a backend

```python
from ailab_vision import Detection, register_backend


@register_backend("my-model")
class MyModel:
    def detect(self, frame) -> list[Detection]: ...
```

No inheritance needed: any class with a matching `detect()` satisfies the `Detector` protocol.

## Design decisions

- **One interface, many backends.** Models change yearly; application code should not.
- **Lazy heavy imports.** PyTorch and OpenCV load only when a feature that needs them is used.
- **Greedy IoU tracking.** No SciPy needed and accurate at normal frame rates. A Kalman-filter
  motion model (SORT-style) is the planned next step for fast objects or low frame rates.
- **Tested without a model.** The YOLO adapter is tested against a fake Ultralytics module,
  so the suite runs in seconds with no downloads or GPU.

## Third-party licences

| Library | Used for | Licence |
|---|---|---|
| NumPy | Arrays | BSD-3-Clause |
| OpenCV | Video I/O, drawing | Apache-2.0 |
| Ultralytics | YOLO backend (optional) | **AGPL-3.0** |

Ultralytics is optional. Using the `yolo` backend in a distributed or networked product brings
AGPL-3.0 obligations, or requires an Ultralytics commercial licence.

## Development

```bash
cd packages/vision
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
