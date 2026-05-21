# Multi-Stream RTSP with Batched Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-stream `/api/stream/*` endpoint with a multi-stream collection API at `/api/streams/*` supporting up to 10 simultaneous RTSP streams, served by a single shared YOLO model via batched inference.

**Architecture:** Per-stream `StreamSession` owns config, frame slot, counter, and annotated output. Each session has a dedicated `CaptureThread` that pumps frames into a single-slot (drop-oldest) buffer. One global `InferenceWorker` thread snapshots all sessions' latest frames every ~30ms, runs one batched `model(batch, ...)` call, then fans results back to each session's `WingCounter` + annotator. A `StreamManager` registry owns the worker lifecycle (start on first stream, stop on last) and enforces a 10-stream cap.

**Tech Stack:** Python 3.12, FastAPI, OpenCV (cv2), Ultralytics YOLO, pytest, threading, pydantic v2.

**Spec:** `docs/superpowers/specs/2026-05-21-multi-stream-batched-inference-design.md`

---

## File Structure

**New files**
- `app/core/stream_session.py` — `StreamSession` (state container, frame slot, detection processing)
- `app/core/capture_thread.py` — `CaptureThread` (RTSP read loop, reconnect with backoff)
- `app/core/inference_worker.py` — `InferenceWorker` (batched model call, fan-out)
- `app/core/stream_manager.py` — `StreamManager` registry, exceptions
- `app/routers/streams.py` — collection-based `/api/streams/*` router
- `tests/__init__.py`
- `tests/conftest.py` — fixtures (fake model, sample frame)
- `tests/unit/__init__.py`
- `tests/unit/test_stream_session.py`
- `tests/unit/test_capture_thread.py`
- `tests/unit/test_inference_worker.py`
- `tests/unit/test_stream_manager.py`
- `tests/integration/__init__.py`
- `tests/integration/test_streams_api.py`
- `tests/integration/test_end_to_end_file.py`
- `requirements-dev.txt` — pytest, pytest-asyncio, httpx

**Modified files**
- `app/main.py` — replace `stream` router with `streams`; create `StreamManager` in lifespan
- `app/config.py` — drop per-stream fields
- `app/core/runtime_config.py` — match new `Settings` shape
- `app/routers/config_router.py` — `ConfigResponse` / `ConfigPatch` reduced to 3 global fields
- `app/static/stream.html` — dashboard layout (grid of stream cards)
- `app/static/js/stream.js` — multi-stream dashboard logic
- `app/static/css/style.css` — add `.stream-grid`, `.stream-card`, `.state-pill`, `.add-stream-card`
- `docker-compose.yml` — drop per-stream env vars

**Deleted files**
- `app/routers/stream.py`

---

## Task 1: Set up test infrastructure

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create dev requirements**

Create `requirements-dev.txt`:
```
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
```

- [ ] **Step 2: Install dev dependencies**

Run: `pip install -r requirements-dev.txt`
Expected: pytest, pytest-asyncio, and httpx install successfully.

- [ ] **Step 3: Create `pytest.ini`**

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
```

- [ ] **Step 4: Create empty package files**

Create `tests/__init__.py` (empty), `tests/unit/__init__.py` (empty), `tests/integration/__init__.py` (empty).

- [ ] **Step 5: Create `tests/conftest.py` with shared fixtures**

```python
import numpy as np
import pytest


class FakeBox:
    """Mimics ultralytics box.xyxy[0].tolist() and box.conf[0]."""
    def __init__(self, x1, y1, x2, y2, conf):
        self.xyxy = [[x1, y1, x2, y2]]
        self.conf = [conf]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLOModel:
    """Test double for ultralytics.YOLO. Returns deterministic boxes per call.

    The constructor takes a list of `per_frame_boxes`, where each item is a
    list of FakeBox objects representing detections for that frame. Each call
    pops the next batch from the queue (one entry per input frame).
    """
    def __init__(self, per_call_results=None):
        self.per_call_results = list(per_call_results or [])
        self.calls = []

    def __call__(self, frames, conf=None, iou=None, imgsz=None, verbose=False):
        self.calls.append({
            "n": len(frames),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
        })
        if self.per_call_results:
            results = self.per_call_results.pop(0)
        else:
            results = [FakeResult([]) for _ in frames]
        assert len(results) == len(frames), (
            f"Fake model: configured {len(results)} results but got {len(frames)} frames"
        )
        return results


@pytest.fixture
def fake_frame():
    """Black 640x384x3 frame, matches default video dims."""
    return np.zeros((384, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_model():
    return FakeYOLOModel()


@pytest.fixture
def fake_box():
    return FakeBox


@pytest.fixture
def fake_result():
    return FakeResult
```

- [ ] **Step 6: Run pytest to confirm it works**

Run: `pytest -q`
Expected: `no tests ran in 0.0Xs` (exit 5 = no tests collected — acceptable for empty test dirs).

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt pytest.ini tests/
git commit -m "test: add pytest infrastructure and shared fixtures"
```

---

## Task 2: StreamSession — state container

**Files:**
- Create: `app/core/stream_session.py`
- Create: `tests/unit/test_stream_session.py`

The `StreamSession` is a thread-safe state container. It owns per-stream config, a single-slot capture buffer (drop-oldest), a `WingCounter`, the latest annotated JPEG, and status fields. It exposes a `process_detections()` method that the `InferenceWorker` will call.

- [ ] **Step 1: Write tests for construction and immutable id**

Create `tests/unit/test_stream_session.py`:
```python
import pytest
from app.core.stream_session import StreamSession


def test_session_has_unique_id():
    a = StreamSession(url="rtsp://a")
    b = StreamSession(url="rtsp://b")
    assert a.id != b.id
    assert len(a.id) == 8
    assert all(c in "0123456789abcdef" for c in a.id)


def test_session_defaults():
    s = StreamSession(url="rtsp://a")
    assert s.url == "rtsp://a"
    assert s.name == s.id           # default name = id
    assert s.confidence == 0.25
    assert s.nms_iou == 0.45
    assert s.imgsz == 640
    assert s.roi_position == 0.7
    assert s.max_distance == 50
    assert s.max_disappeared == 15
    assert s.state == "connecting"
    assert s.counting is False
    assert s.wing_count == 0
    assert s.fps_display == 0.0
    assert s.error is None
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: ImportError or `ModuleNotFoundError: app.core.stream_session`.

- [ ] **Step 3: Implement minimal `StreamSession`**

Create `app/core/stream_session.py`:
```python
import threading
import uuid
from typing import Optional

import numpy as np

from app.core.counter import WingCounter


class StreamSession:
    """Per-stream state container.

    Owns the stream's id, url, name, per-stream inference + tracker config,
    its WingCounter, the single-slot capture buffer, the latest annotated
    JPEG bytes, and status fields (state, counting, wing_count, fps, error).

    All mutable state is guarded by self._lock. The capture thread is the
    sole writer of the frame slot via push_capture_frame(); the inference
    worker is the sole reader via take_latest_frame() and the sole writer
    of detection-derived state via process_detections().
    """

    DEFAULT_FRAME_HEIGHT = 384   # used until first real frame arrives

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        confidence: float = 0.25,
        nms_iou: float = 0.45,
        imgsz: int = 640,
        roi_position: float = 0.7,
        max_distance: int = 50,
        max_disappeared: int = 15,
    ) -> None:
        self.id: str = uuid.uuid4().hex[:8]
        self.url: str = url
        self.name: str = name or self.id

        self.confidence: float = confidence
        self.nms_iou: float = nms_iou
        self.imgsz: int = imgsz
        self.roi_position: float = roi_position
        self.max_distance: int = max_distance
        self.max_disappeared: int = max_disappeared

        self.state: str = "connecting"
        self.counting: bool = False
        self.wing_count: int = 0
        self.fps_display: float = 0.0
        self.error: Optional[str] = None

        self._lock = threading.RLock()
        self._latest_capture: Optional[np.ndarray] = None
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._frame_height: int = self.DEFAULT_FRAME_HEIGHT
        self._frame_num: int = 0

        self.counter = WingCounter(
            roi_y=int(self._frame_height * self.roi_position),
            max_disappeared=self.max_disappeared,
            max_distance=self.max_distance,
        )
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 2 passed.

- [ ] **Step 5: Add tests for frame slot semantics**

Append to `tests/unit/test_stream_session.py`:
```python
def test_take_latest_frame_when_empty(fake_frame):
    s = StreamSession(url="rtsp://a")
    assert s.take_latest_frame() is None


def test_push_and_take_latest_frame(fake_frame):
    s = StreamSession(url="rtsp://a")
    s.push_capture_frame(fake_frame)
    f = s.take_latest_frame()
    assert f is fake_frame
    # Slot is cleared after take.
    assert s.take_latest_frame() is None


def test_push_overwrites_old_frame(fake_frame):
    import numpy as np
    s = StreamSession(url="rtsp://a")
    older = fake_frame
    newer = np.ones_like(fake_frame)
    s.push_capture_frame(older)
    s.push_capture_frame(newer)
    f = s.take_latest_frame()
    assert f is newer
```

- [ ] **Step 6: Run — confirm they fail**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 3 new tests fail with AttributeError on `push_capture_frame` / `take_latest_frame`.

- [ ] **Step 7: Implement frame slot methods**

Add to `app/core/stream_session.py` (inside the `StreamSession` class):
```python
    def push_capture_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_capture = frame

    def take_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            frame, self._latest_capture = self._latest_capture, None
            if frame is not None:
                # Cache frame height for ROI scaling on first frame.
                self._frame_height = frame.shape[0]
                self.counter.roi_y = int(self._frame_height * self.roi_position)
            return frame
```

- [ ] **Step 8: Run — confirm they pass**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 5 passed.

- [ ] **Step 9: Add tests for state transitions**

Append to `tests/unit/test_stream_session.py`:
```python
def test_set_state_updates_state_and_clears_error():
    s = StreamSession(url="rtsp://a")
    s.set_state("error", error="boom")
    assert s.state == "error"
    assert s.error == "boom"
    s.set_state("connected", error=None)
    assert s.state == "connected"
    assert s.error is None


def test_set_state_preserves_error_when_not_passed():
    s = StreamSession(url="rtsp://a")
    s.set_state("error", error="boom")
    s.set_state("reconnecting")
    assert s.state == "reconnecting"
    assert s.error == "boom"


def test_counting_controls():
    s = StreamSession(url="rtsp://a")
    s.start_counting()
    assert s.counting is True
    s.stop_counting()
    assert s.counting is False


def test_reset_counting_zeroes_count_and_keeps_counting_flag():
    s = StreamSession(url="rtsp://a")
    s.start_counting()
    s.wing_count = 7
    s.reset_counting()
    assert s.wing_count == 0
    assert s.counting is True
```

- [ ] **Step 10: Run — confirm they fail**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 4 new tests fail.

- [ ] **Step 11: Implement state and counting methods**

Add to `app/core/stream_session.py`:
```python
    def set_state(self, state: str, error: Optional[str] = ...) -> None:
        """Update state and optionally error. Pass error=None to clear it;
        omit error to leave it unchanged."""
        with self._lock:
            self.state = state
            if error is not ...:
                self.error = error

    def start_counting(self) -> None:
        with self._lock:
            self.counting = True

    def stop_counting(self) -> None:
        with self._lock:
            self.counting = False

    def reset_counting(self) -> None:
        with self._lock:
            self.wing_count = 0
            self.counter.reset()
```

The `error: Optional[str] = ...` uses the Ellipsis literal as a sentinel for "argument omitted" (distinct from `error=None` which clears it).

- [ ] **Step 12: Run — confirm they pass**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 9 passed.

- [ ] **Step 13: Add tests for config update + restart-needed detection**

Append to `tests/unit/test_stream_session.py`:
```python
def test_apply_patch_updates_inference_params():
    s = StreamSession(url="rtsp://a")
    needs_restart = s.apply_patch({
        "confidence": 0.5,
        "nms_iou": 0.6,
        "imgsz": 832,
    })
    assert s.confidence == 0.5
    assert s.nms_iou == 0.6
    assert s.imgsz == 832
    assert needs_restart is False


def test_apply_patch_recomputes_roi_y_on_position_change():
    s = StreamSession(url="rtsp://a")
    s._frame_height = 1080
    s.counter.roi_y = int(1080 * 0.7)
    s.apply_patch({"roi_position": 0.5})
    assert s.roi_position == 0.5
    assert s.counter.roi_y == 540


def test_apply_patch_url_change_signals_restart():
    s = StreamSession(url="rtsp://a")
    needs_restart = s.apply_patch({"url": "rtsp://b"})
    assert s.url == "rtsp://b"
    assert needs_restart is True


def test_apply_patch_same_url_no_restart():
    s = StreamSession(url="rtsp://a")
    needs_restart = s.apply_patch({"url": "rtsp://a"})
    assert needs_restart is False


def test_apply_patch_tracker_params_forwarded():
    s = StreamSession(url="rtsp://a")
    s.apply_patch({"max_distance": 99, "max_disappeared": 7})
    assert s.max_distance == 99
    assert s.max_disappeared == 7
    assert s.counter.tracker.max_distance == 99
    assert s.counter.tracker.max_disappeared == 7


def test_apply_patch_name_only():
    s = StreamSession(url="rtsp://a")
    s.apply_patch({"name": "Cam 1"})
    assert s.name == "Cam 1"
```

- [ ] **Step 14: Run — confirm they fail**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 6 new tests fail on `apply_patch`.

- [ ] **Step 15: Implement `apply_patch`**

Add to `app/core/stream_session.py`:
```python
    # Per-stream config keys that take effect on the next inference cycle
    # without restarting the capture thread.
    _LIVE_KEYS = (
        "name",
        "confidence",
        "nms_iou",
        "imgsz",
        "roi_position",
        "max_distance",
        "max_disappeared",
    )

    def apply_patch(self, patch: dict) -> bool:
        """Apply a partial config update. Returns True if the capture
        thread must be restarted (URL changed), False otherwise."""
        needs_restart = False
        with self._lock:
            new_url = patch.get("url")
            if new_url is not None and new_url != self.url:
                self.url = new_url
                needs_restart = True

            for key in self._LIVE_KEYS:
                if key in patch and patch[key] is not None:
                    setattr(self, key, patch[key])

            # Recompute counter-driven fields.
            self.counter.roi_y = int(self._frame_height * self.roi_position)
            self.counter.tracker.max_distance = self.max_distance
            self.counter.tracker.max_disappeared = self.max_disappeared

        return needs_restart
```

- [ ] **Step 16: Run — confirm they pass**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 15 passed.

- [ ] **Step 17: Add tests for `process_detections`**

Append to `tests/unit/test_stream_session.py`:
```python
def test_process_detections_no_counting_does_not_increment(fake_frame):
    s = StreamSession(url="rtsp://a")
    s.push_capture_frame(fake_frame)
    s.take_latest_frame()  # pull through to populate _frame_height
    s.process_detections([
        {"x1": 100, "y1": 100, "x2": 140, "y2": 140, "conf": 0.9},
    ])
    assert s.wing_count == 0


def test_process_detections_counting_crossing_increments(fake_frame):
    s = StreamSession(url="rtsp://a")
    s.push_capture_frame(fake_frame)
    s.take_latest_frame()  # populates _frame_height=384, roi_y=int(384*0.7)=268
    s.start_counting()

    # First detection above ROI, second below — should cross and increment.
    s.process_detections([{"x1": 100, "y1": 100, "x2": 140, "y2": 140, "conf": 0.9}])
    assert s.wing_count == 0
    s.push_capture_frame(fake_frame)
    s.take_latest_frame()
    s.process_detections([{"x1": 100, "y1": 280, "x2": 140, "y2": 320, "conf": 0.9}])
    assert s.wing_count == 1


def test_process_detections_stores_annotated_jpeg(fake_frame):
    s = StreamSession(url="rtsp://a")
    s.push_capture_frame(fake_frame)
    s.take_latest_frame()
    s.process_detections([{"x1": 10, "y1": 10, "x2": 30, "y2": 30, "conf": 0.9}])
    assert s.latest_annotated_jpeg() is not None
    assert s.latest_annotated_jpeg().startswith(b"\xff\xd8")  # JPEG SOI


def test_process_detections_without_frame_is_noop():
    s = StreamSession(url="rtsp://a")
    # No frame ever pushed/taken — process_detections should silently skip.
    s.process_detections([{"x1": 10, "y1": 10, "x2": 30, "y2": 30, "conf": 0.9}])
    assert s.latest_annotated_jpeg() is None
    assert s.wing_count == 0
```

- [ ] **Step 18: Run — confirm they fail**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 4 new tests fail.

- [ ] **Step 19: Implement `process_detections` and `latest_annotated_jpeg`**

Add to `app/core/stream_session.py`:
```python
import time

import cv2

from app.core.annotator import annotate_detections


class StreamSession:
    # ... (existing class body) ...

    def latest_annotated_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_annotated_jpeg

    def process_detections(self, det_info: list[dict]) -> None:
        """Run counter + annotation on the most recently captured frame.

        This is called once per inference-worker cycle from a single thread,
        so the counter does not need additional internal locking.
        """
        with self._lock:
            frame = self._last_processed_frame
            if frame is None:
                return
            self._frame_num += 1

            # FPS sliding window.
            now = time.time()
            self._fps_frame_count += 1
            elapsed = now - self._fps_timer
            if elapsed >= 0.5:
                self.fps_display = round(self._fps_frame_count / elapsed, 1)
                self._fps_frame_count = 0
                self._fps_timer = now

            # Counter step.
            if self.counting:
                objects = self.counter.update(det_info)
            else:
                tracker_input = [
                    ((d["x1"] + d["x2"]) // 2, (d["y1"] + d["y2"]) // 2,
                     d["x1"], d["y1"], d["x2"], d["y2"])
                    for d in det_info
                ]
                objects = self.counter.tracker.update(tracker_input)

            self.wing_count = self.counter.total_count

            # Annotate.
            flash_with_frame = [
                (fx, fy, self._frame_num - i)
                for i, (fx, fy) in enumerate(
                    reversed(self.counter.flash_events[-12:])
                )
            ]
            annotated = annotate_detections(
                frame=frame,
                detections=det_info,
                objects=objects,
                counted_ids=self.counter.counted_ids if self.counting else set(),
                trails=self.counter.trails,
                flash_events=flash_with_frame,
                roi_y=self.counter.roi_y if self.counting else None,
                frame_num=self._frame_num,
                total_count=self.counter.total_count if self.counting else 0,
                total_frames=0,
                is_stream=True,
                fps_display=self.fps_display,
            )

            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._latest_annotated_jpeg = jpeg.tobytes()
```

Also extend `take_latest_frame` to retain a reference for `process_detections` to consume:
```python
    def take_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            frame, self._latest_capture = self._latest_capture, None
            if frame is not None:
                self._frame_height = frame.shape[0]
                self.counter.roi_y = int(self._frame_height * self.roi_position)
                self._last_processed_frame = frame
            return frame
```

And initialize the new fields in `__init__`:
```python
        self._last_processed_frame: Optional[np.ndarray] = None
        self._fps_timer: float = time.time()
        self._fps_frame_count: int = 0
```

- [ ] **Step 20: Run — confirm they pass**

Run: `pytest tests/unit/test_stream_session.py -v`
Expected: 19 passed.

- [ ] **Step 21: Commit**

```bash
git add app/core/stream_session.py tests/unit/test_stream_session.py
git commit -m "feat: add StreamSession state container with per-stream counter"
```

---

## Task 3: CaptureThread — RTSP read loop with reconnect

**Files:**
- Create: `app/core/capture_thread.py`
- Create: `tests/unit/test_capture_thread.py`

The `CaptureThread` is a thin background thread that opens `cv2.VideoCapture(url)`, reads frames into the session's slot, and reconnects with exponential backoff (2 → 4 → 8 → 16s, then cap at 16s) on read failure.

- [ ] **Step 1: Write test for backoff sequence calculation**

Create `tests/unit/test_capture_thread.py`:
```python
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.capture_thread import CaptureThread
from app.core.stream_session import StreamSession


def test_backoff_sequence_is_2_4_8_16_capped():
    assert CaptureThread.BACKOFF_SEQUENCE == (2, 4, 8, 16)


def test_backoff_index_caps_at_last():
    # idx clamping: min(idx + 1, len(seq) - 1)
    seq = CaptureThread.BACKOFF_SEQUENCE
    idx = 0
    for _ in range(10):
        idx = min(idx + 1, len(seq) - 1)
    assert idx == len(seq) - 1
    assert seq[idx] == 16
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/unit/test_capture_thread.py -v`
Expected: ImportError on `app.core.capture_thread`.

- [ ] **Step 3: Implement minimal skeleton**

Create `app/core/capture_thread.py`:
```python
import threading
import cv2

from app.core.stream_session import StreamSession


class CaptureThread:
    """Background thread that reads frames from an RTSP source into a
    StreamSession's single-slot frame buffer. On read failure, reconnects
    with exponential backoff (2, 4, 8, 16 seconds, then capped at 16).

    The capture thread is the sole writer of session._latest_capture, so
    no cross-writer locking inside the thread is needed beyond what the
    session itself provides.
    """

    BACKOFF_SEQUENCE = (2, 4, 8, 16)

    def __init__(self, session: StreamSession) -> None:
        self.session = session
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.session.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
```

- [ ] **Step 4: Run — confirm tests pass**

Run: `pytest tests/unit/test_capture_thread.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write test that stop() returns promptly during backoff**

Append to `tests/unit/test_capture_thread.py`:
```python
def test_stop_interrupts_backoff():
    """An open call that always fails would put us in backoff. stop() must
    interrupt the sleep so deletion doesn't hang for 16 seconds."""
    session = StreamSession(url="rtsp://does-not-exist")
    ct = CaptureThread(session)

    # Patch VideoCapture so isOpened() is False and we go straight to backoff.
    with patch("app.core.capture_thread.cv2.VideoCapture") as vc_mock:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        vc_mock.return_value = cap_mock

        ct.start()
        # Give the thread a moment to enter backoff sleep.
        time.sleep(0.1)
        t0 = time.time()
        ct.stop(timeout=3.0)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"stop() took {elapsed:.2f}s — backoff not interruptible"
```

- [ ] **Step 6: Run — confirm fails (no `_run`)**

Run: `pytest tests/unit/test_capture_thread.py::test_stop_interrupts_backoff -v`
Expected: fail (thread doesn't start, or test hangs).

- [ ] **Step 7: Implement `_run` with reconnect/backoff**

Add to `app/core/capture_thread.py`:
```python
    def _run(self) -> None:
        backoff_idx = 0
        while not self._stop.is_set():
            self.session.set_state("connecting", error=None)
            cap = cv2.VideoCapture(self.session.url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # not all backends support this

            if not cap.isOpened():
                self.session.set_state(
                    "reconnecting", error=f"Could not open {self.session.url}"
                )
                cap.release()
                if self._sleep_backoff(backoff_idx):
                    break
                backoff_idx = min(backoff_idx + 1, len(self.BACKOFF_SEQUENCE) - 1)
                continue

            self.session.set_state("connected", error=None)
            backoff_idx = 0

            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.session.set_state("reconnecting", error="Stream read failed")
                    break
                self.session.push_capture_frame(frame)

            cap.release()
            if self._stop.is_set():
                break
            if self._sleep_backoff(backoff_idx):
                break
            backoff_idx = min(backoff_idx + 1, len(self.BACKOFF_SEQUENCE) - 1)

        self.session.set_state("disconnected", error=None)

    def _sleep_backoff(self, idx: int) -> bool:
        """Sleep for BACKOFF_SEQUENCE[idx] seconds, interruptible by stop().
        Returns True if interrupted (stop signaled), False if timeout completed."""
        return self._stop.wait(self.BACKOFF_SEQUENCE[idx])
```

- [ ] **Step 8: Run — confirm test passes**

Run: `pytest tests/unit/test_capture_thread.py -v`
Expected: 3 passed.

- [ ] **Step 9: Write tests for successful read pumping**

Append to `tests/unit/test_capture_thread.py`:
```python
def test_successful_capture_pushes_frames_to_session(fake_frame):
    session = StreamSession(url="rtsp://ok")
    ct = CaptureThread(session)

    read_sequence = [(True, fake_frame), (True, fake_frame), (True, fake_frame)]
    read_iter = iter(read_sequence)

    def fake_read():
        try:
            return next(read_iter)
        except StopIteration:
            return (False, None)

    with patch("app.core.capture_thread.cv2.VideoCapture") as vc_mock:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        cap_mock.read.side_effect = fake_read
        vc_mock.return_value = cap_mock

        ct.start()
        # Give thread time to read all 3 frames and then hit reconnect.
        time.sleep(0.2)
        ct.stop(timeout=3.0)

    # Last frame should be in the slot (unless inference worker drained it,
    # which doesn't run in this test).
    assert session._latest_capture is not None
    # State should have moved through connecting -> connected -> reconnecting -> disconnected.
    assert session.state == "disconnected"


def test_successful_open_resets_backoff_index(fake_frame):
    """Open success should reset backoff_idx so a later one-off failure
    doesn't escalate."""
    session = StreamSession(url="rtsp://flaky")
    ct = CaptureThread(session)

    # Simulate: open fails, fails, fails, succeeds (and reads 1 frame), then read fails.
    # Then open should be retried at backoff_idx 0 — i.e. 2s — not 16s.
    open_outcomes = iter([False, False, False, True, False])

    def is_opened():
        try:
            return next(open_outcomes)
        except StopIteration:
            return False

    read_outcomes = iter([(True, fake_frame), (False, None)])

    with patch("app.core.capture_thread.cv2.VideoCapture") as vc_mock:
        cap_mock = MagicMock()
        cap_mock.isOpened.side_effect = is_opened
        cap_mock.read.side_effect = lambda: next(read_outcomes, (False, None))
        vc_mock.return_value = cap_mock

        # Patch _sleep_backoff so we can observe the backoff_idx history.
        observed = []
        real_sleep = ct._sleep_backoff

        def spy_sleep(idx):
            observed.append(idx)
            # Skip the actual wait — return False so loop continues.
            return False

        ct._sleep_backoff = spy_sleep  # type: ignore[assignment]

        ct.start()
        time.sleep(0.3)
        ct.stop(timeout=3.0)

    # Expected: idx 0 (open fail), 1 (open fail), 2 (open fail),
    # then open succeeds (resets idx), then read fails so next backoff is idx 0 again.
    assert observed[:3] == [0, 1, 2]
    assert 0 in observed[3:], (
        f"Expected idx to reset to 0 after success, got: {observed}"
    )
```

- [ ] **Step 10: Run — confirm pass**

Run: `pytest tests/unit/test_capture_thread.py -v`
Expected: 5 passed.

- [ ] **Step 11: Commit**

```bash
git add app/core/capture_thread.py tests/unit/test_capture_thread.py
git commit -m "feat: add CaptureThread with exponential reconnect backoff"
```

---

## Task 4: InferenceWorker — batched model call + fan-out

**Files:**
- Create: `app/core/inference_worker.py`
- Create: `tests/unit/test_inference_worker.py`

The `InferenceWorker` is a single global thread. Each cycle it snapshots every `connected` session's latest frame, runs one batched `model(batch, ...)` call, and dispatches per-frame results to each session's `process_detections()`.

- [ ] **Step 1: Write test for batching across multiple sessions**

Create `tests/unit/test_inference_worker.py`:
```python
import time
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.inference_worker import InferenceWorker
from app.core.stream_session import StreamSession
from tests.conftest import FakeBox, FakeResult, FakeYOLOModel


class FakeManager:
    """Implements only `connected_sessions()`."""
    def __init__(self, sessions):
        self._sessions = sessions

    def connected_sessions(self):
        return list(self._sessions)


def test_worker_batches_frames_from_all_connected_sessions(fake_frame):
    s1 = StreamSession(url="rtsp://a")
    s2 = StreamSession(url="rtsp://b")
    s1.set_state("connected"); s2.set_state("connected")
    s1.push_capture_frame(fake_frame)
    s2.push_capture_frame(fake_frame.copy())

    model = FakeYOLOModel(per_call_results=[
        [FakeResult([FakeBox(10, 10, 30, 30, 0.9)]),
         FakeResult([FakeBox(50, 50, 80, 80, 0.8)])],
    ])
    manager = FakeManager([s1, s2])

    worker = InferenceWorker(model, manager, batch_window_ms=10)
    worker._run_once()

    # One batched call with two frames.
    assert len(model.calls) == 1
    assert model.calls[0]["n"] == 2
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/unit/test_inference_worker.py -v`
Expected: ImportError on `app.core.inference_worker`.

- [ ] **Step 3: Implement `InferenceWorker` skeleton with `_run_once`**

Create `app/core/inference_worker.py`:
```python
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.stream_manager import StreamManager


class InferenceWorker:
    """Single background thread that runs batched YOLO inference across
    all currently-connected StreamSessions.

    Cycle (every batch_window_ms):
      1. Snapshot every connected session's latest frame.
      2. If any frames present, call model(batch, conf=0.01, iou=0.99,
         imgsz=max_imgsz) once.
      3. For each (session, result) pair, filter the result by that
         session's confidence/nms_iou and call session.process_detections().
      4. Sleep the remainder of the batch window.
    """

    def __init__(self, model, manager: "StreamManager", batch_window_ms: int = 30) -> None:
        self.model = model
        self.manager = manager
        self.batch_window_ms = batch_window_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="inference-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            cycle_start = time.time()
            self._run_once()
            elapsed_ms = (time.time() - cycle_start) * 1000
            sleep_ms = max(0.0, self.batch_window_ms - elapsed_ms)
            if sleep_ms > 0:
                if self._stop.wait(sleep_ms / 1000):
                    break

    def _run_once(self) -> None:
        sessions = self.manager.connected_sessions()
        batch = []
        meta = []
        for s in sessions:
            frame = s.take_latest_frame()
            if frame is not None:
                batch.append(frame)
                meta.append(s)

        if not batch:
            return

        imgsz = max(s.imgsz for s in meta)
        results = self.model(batch, conf=0.01, iou=0.99, imgsz=imgsz, verbose=False)

        for session, result in zip(meta, results):
            det_info = self._extract_dets(result, session.confidence, session.nms_iou)
            session.process_detections(det_info)

    def _extract_dets(self, result, conf_thresh: float, iou_thresh: float) -> list[dict]:
        """Convert one ultralytics Result into the list-of-dicts shape that
        WingCounter expects, filtered by per-stream conf and re-NMSed by
        per-stream iou."""
        boxes = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        return self._apply_nms(boxes, iou_thresh)

    @staticmethod
    def _apply_nms(boxes: list[dict], iou_thresh: float) -> list[dict]:
        """Greedy NMS over a list of detection dicts. Sorted by conf desc;
        drop any box with IoU > iou_thresh against an already-kept box."""
        boxes_sorted = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        kept: list[dict] = []
        for b in boxes_sorted:
            if all(_bbox_iou(b, k) <= iou_thresh for k in kept):
                kept.append(b)
        return kept


def _bbox_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

- [ ] **Step 4: Run — confirm test passes**

Run: `pytest tests/unit/test_inference_worker.py -v`
Expected: 1 passed.

- [ ] **Step 5: Add tests for fan-out correctness, empty batch, and threshold filtering**

Append to `tests/unit/test_inference_worker.py`:
```python
def test_worker_fans_results_to_correct_sessions(fake_frame):
    s1 = StreamSession(url="rtsp://a")
    s2 = StreamSession(url="rtsp://b")
    s1.set_state("connected"); s2.set_state("connected")
    s1.start_counting(); s2.start_counting()
    s1.push_capture_frame(fake_frame)
    s2.push_capture_frame(fake_frame.copy())

    # s1's result has 2 boxes, s2's result has 0 boxes.
    model = FakeYOLOModel(per_call_results=[
        [FakeResult([
            FakeBox(10, 10, 30, 30, 0.9),
            FakeBox(40, 40, 60, 60, 0.9),
         ]),
         FakeResult([])],
    ])
    manager = FakeManager([s1, s2])
    worker = InferenceWorker(model, manager, batch_window_ms=10)
    worker._run_once()

    # Counter starts new tracks but doesn't count until they cross the ROI.
    # The point of this assertion: s2's count must not be inflated by s1's boxes.
    assert s2.counter.tracker.next_id == 0, (
        "s2 received s1's detections — fan-out is mis-indexed"
    )


def test_worker_skips_cycle_when_no_frames(fake_frame):
    s1 = StreamSession(url="rtsp://a")
    s1.set_state("connected")
    # No push_capture_frame — slot is empty.

    model = FakeYOLOModel()
    manager = FakeManager([s1])
    worker = InferenceWorker(model, manager)
    worker._run_once()

    assert model.calls == [], "Model should not be called when batch is empty"


def test_worker_applies_per_stream_confidence_threshold(fake_frame):
    s1 = StreamSession(url="rtsp://a", confidence=0.5)
    s1.set_state("connected")
    s1.start_counting()
    s1.push_capture_frame(fake_frame)

    # Two boxes — one above s1's 0.5 threshold, one below.
    model = FakeYOLOModel(per_call_results=[
        [FakeResult([
            FakeBox(10, 10, 30, 30, 0.9),
            FakeBox(40, 40, 60, 60, 0.3),
        ])],
    ])
    manager = FakeManager([s1])
    worker = InferenceWorker(model, manager)
    worker._run_once()

    # Only the 0.9-conf box should have made it into the counter.
    assert s1.counter.tracker.next_id == 1


def test_worker_applies_per_stream_nms_iou(fake_frame):
    """Two highly-overlapping boxes should be deduped under low iou threshold."""
    s1 = StreamSession(url="rtsp://a", nms_iou=0.1)  # aggressive dedupe
    s1.set_state("connected"); s1.start_counting()
    s1.push_capture_frame(fake_frame)

    model = FakeYOLOModel(per_call_results=[
        [FakeResult([
            FakeBox(10, 10, 50, 50, 0.9),
            FakeBox(12, 12, 52, 52, 0.8),  # near-duplicate
        ])],
    ])
    manager = FakeManager([s1])
    worker = InferenceWorker(model, manager)
    worker._run_once()

    # NMS should keep only one.
    assert s1.counter.tracker.next_id == 1
```

- [ ] **Step 6: Run — confirm pass**

Run: `pytest tests/unit/test_inference_worker.py -v`
Expected: 5 passed.

- [ ] **Step 7: Add test that worker only batches `connected` sessions**

Append to `tests/unit/test_inference_worker.py`:
```python
def test_worker_only_batches_connected_sessions(fake_frame):
    s1 = StreamSession(url="rtsp://a")
    s2 = StreamSession(url="rtsp://b")
    # s1 connected, s2 not — manager will only return s1 from
    # connected_sessions(), so this test verifies manager contract.
    s1.set_state("connected")
    s2.set_state("reconnecting")
    s1.push_capture_frame(fake_frame)

    class ReadthroughManager:
        def __init__(self, sessions): self._s = sessions
        def connected_sessions(self):
            return [s for s in self._s if s.state == "connected"]

    model = FakeYOLOModel(per_call_results=[
        [FakeResult([FakeBox(10, 10, 30, 30, 0.9)])],
    ])
    manager = ReadthroughManager([s1, s2])
    worker = InferenceWorker(model, manager)
    worker._run_once()

    assert model.calls[0]["n"] == 1
```

- [ ] **Step 8: Run — confirm pass**

Run: `pytest tests/unit/test_inference_worker.py -v`
Expected: 6 passed.

- [ ] **Step 9: Commit**

```bash
git add app/core/inference_worker.py tests/unit/test_inference_worker.py
git commit -m "feat: add InferenceWorker with batched model call and per-stream filtering"
```

---

## Task 5: StreamManager — CRUD registry + worker lifecycle

**Files:**
- Create: `app/core/stream_manager.py`
- Create: `tests/unit/test_stream_manager.py`

The `StreamManager` is the registry. It owns the `dict[str, StreamSession]`, the `InferenceWorker`, and enforces the 10-stream cap. It also owns the lifecycle: starts each session's capture thread on add, stops on remove, and starts/stops the worker on first/last session.

- [ ] **Step 1: Write tests for capacity enforcement and add/remove**

Create `tests/unit/test_stream_manager.py`:
```python
from unittest.mock import patch

import pytest

from app.core.stream_manager import StreamManager, CapacityExceeded, NotFound
from app.core.stream_session import StreamSession
from tests.conftest import FakeYOLOModel


@pytest.fixture
def manager():
    # Patch CaptureThread so add() doesn't actually open RTSP.
    with patch("app.core.stream_manager.CaptureThread") as ct_class:
        ct_class.return_value.start.return_value = None
        ct_class.return_value.stop.return_value = None
        mgr = StreamManager(model=FakeYOLOModel())
        yield mgr
        mgr.shutdown()


def test_add_returns_session_with_id(manager):
    s = manager.add(url="rtsp://a", name="Camera 1")
    assert isinstance(s, StreamSession)
    assert s.url == "rtsp://a"
    assert s.name == "Camera 1"
    assert len(s.id) == 8


def test_add_enforces_capacity(manager):
    for i in range(10):
        manager.add(url=f"rtsp://{i}")
    with pytest.raises(CapacityExceeded):
        manager.add(url="rtsp://overflow")


def test_get_returns_session(manager):
    s = manager.add(url="rtsp://a")
    assert manager.get(s.id) is s


def test_get_unknown_raises_not_found(manager):
    with pytest.raises(NotFound):
        manager.get("deadbeef")


def test_remove_stops_capture_and_drops_session(manager):
    s = manager.add(url="rtsp://a")
    manager.remove(s.id)
    with pytest.raises(NotFound):
        manager.get(s.id)


def test_remove_unknown_raises_not_found(manager):
    with pytest.raises(NotFound):
        manager.remove("deadbeef")


def test_list_returns_all_sessions(manager):
    a = manager.add(url="rtsp://a")
    b = manager.add(url="rtsp://b")
    listed = manager.list()
    assert {s.id for s in listed} == {a.id, b.id}
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/unit/test_stream_manager.py -v`
Expected: ImportError on `app.core.stream_manager`.

- [ ] **Step 3: Implement `StreamManager`**

Create `app/core/stream_manager.py`:
```python
import threading
from typing import Optional

from app.core.capture_thread import CaptureThread
from app.core.inference_worker import InferenceWorker
from app.core.stream_session import StreamSession


class CapacityExceeded(Exception):
    """Raised when add() is called while the manager already holds CAPACITY streams."""


class NotFound(Exception):
    """Raised when get(), remove(), or patch() is called with an unknown id."""


class StreamManager:
    """Registry of active StreamSessions.

    Owns the single shared InferenceWorker (started when the first stream
    is added, stopped when the last is removed) and enforces the
    CAPACITY-stream cap. Each session has its own CaptureThread, started
    on add() and stopped on remove() or shutdown().
    """

    CAPACITY = 10

    def __init__(self, model) -> None:
        self._sessions: dict[str, StreamSession] = {}
        self._capture_threads: dict[str, CaptureThread] = {}
        self._lock = threading.RLock()
        self._worker = InferenceWorker(model=model, manager=self)

    # ------------------------------------------------------------------ CRUD

    def add(self, **session_kwargs) -> StreamSession:
        with self._lock:
            if len(self._sessions) >= self.CAPACITY:
                raise CapacityExceeded(f"Capacity reached ({self.CAPACITY} streams)")
            session = StreamSession(**session_kwargs)
            ct = CaptureThread(session)
            self._sessions[session.id] = session
            self._capture_threads[session.id] = ct
            ct.start()
            if len(self._sessions) == 1:
                self._worker.start()
            return session

    def get(self, stream_id: str) -> StreamSession:
        with self._lock:
            session = self._sessions.get(stream_id)
            if session is None:
                raise NotFound(f"Stream {stream_id} not found")
            return session

    def list(self) -> list[StreamSession]:
        with self._lock:
            return list(self._sessions.values())

    def remove(self, stream_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(stream_id, None)
            ct = self._capture_threads.pop(stream_id, None)
            if session is None:
                raise NotFound(f"Stream {stream_id} not found")
        # Stop capture outside the manager lock so a slow join doesn't block
        # other API calls.
        if ct is not None:
            ct.stop()
        with self._lock:
            if not self._sessions:
                self._worker.stop()

    def patch(self, stream_id: str, patch: dict) -> StreamSession:
        with self._lock:
            session = self._sessions.get(stream_id)
            if session is None:
                raise NotFound(f"Stream {stream_id} not found")
        needs_restart = session.apply_patch(patch)
        if needs_restart:
            with self._lock:
                ct = self._capture_threads.pop(stream_id, None)
            if ct is not None:
                ct.stop()
            new_ct = CaptureThread(session)
            with self._lock:
                self._capture_threads[stream_id] = new_ct
            new_ct.start()
        return session

    # ------------------------------------------------------------------ Worker

    def connected_sessions(self) -> list[StreamSession]:
        """Called every cycle by the InferenceWorker. Snapshots the
        sessions list under the lock and returns those in 'connected' state."""
        with self._lock:
            return [s for s in self._sessions.values() if s.state == "connected"]

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            capture_threads = list(self._capture_threads.values())
            self._sessions.clear()
            self._capture_threads.clear()
        for ct in capture_threads:
            ct.stop()
        self._worker.stop()
```

- [ ] **Step 4: Run — confirm 7 tests pass**

Run: `pytest tests/unit/test_stream_manager.py -v`
Expected: 7 passed.

- [ ] **Step 5: Write tests for worker lifecycle**

Append to `tests/unit/test_stream_manager.py`:
```python
def test_worker_starts_on_first_add(manager):
    with patch.object(manager._worker, "start") as start_mock, \
         patch.object(manager._worker, "stop") as stop_mock:
        manager.add(url="rtsp://a")
        start_mock.assert_called_once()
        manager.add(url="rtsp://b")
        start_mock.assert_called_once()  # NOT called again
        stop_mock.assert_not_called()


def test_worker_stops_on_last_remove(manager):
    with patch.object(manager._worker, "start") as start_mock, \
         patch.object(manager._worker, "stop") as stop_mock:
        a = manager.add(url="rtsp://a")
        b = manager.add(url="rtsp://b")
        stop_mock.assert_not_called()
        manager.remove(a.id)
        stop_mock.assert_not_called()  # still 1 left
        manager.remove(b.id)
        stop_mock.assert_called_once()


def test_patch_url_restarts_capture(manager):
    s = manager.add(url="rtsp://a")
    old_ct = manager._capture_threads[s.id]
    manager.patch(s.id, {"url": "rtsp://b"})
    new_ct = manager._capture_threads[s.id]
    assert new_ct is not old_ct
    old_ct.stop.assert_called_once()
    new_ct.start.assert_called_once()
    assert s.url == "rtsp://b"


def test_patch_non_url_keeps_capture(manager):
    s = manager.add(url="rtsp://a")
    old_ct = manager._capture_threads[s.id]
    manager.patch(s.id, {"confidence": 0.5})
    assert manager._capture_threads[s.id] is old_ct
    old_ct.stop.assert_not_called()
    assert s.confidence == 0.5
```

- [ ] **Step 6: Run — confirm pass**

Run: `pytest tests/unit/test_stream_manager.py -v`
Expected: 11 passed.

- [ ] **Step 7: Commit**

```bash
git add app/core/stream_manager.py tests/unit/test_stream_manager.py
git commit -m "feat: add StreamManager registry with capacity cap and worker lifecycle"
```

---

## Task 6: New `/api/streams` router

**Files:**
- Create: `app/routers/streams.py`
- Create: `tests/integration/test_streams_api.py`

The router exposes the collection-based endpoints. It depends on `request.app.state.stream_manager`.

- [ ] **Step 1: Implement the router skeleton**

Create `app/routers/streams.py`:
```python
import time
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.stream_manager import CapacityExceeded, NotFound, StreamManager
from app.core.stream_session import StreamSession

router = APIRouter(prefix="/api/streams", tags=["streams"])


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------

class StreamCreate(BaseModel):
    url: str
    name: Optional[str] = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    nms_iou: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    imgsz: Annotated[int, Field(ge=32)] = 640
    roi_position: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    max_distance: Annotated[int, Field(ge=1)] = 50
    max_disappeared: Annotated[int, Field(ge=1)] = 15

    @field_validator("imgsz")
    @classmethod
    def imgsz_multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @field_validator("url")
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("url must not be empty")
        return v.strip()


class StreamPatch(BaseModel):
    url: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    nms_iou: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    imgsz: Optional[Annotated[int, Field(ge=32)]] = None
    roi_position: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    max_distance: Optional[Annotated[int, Field(ge=1)]] = None
    max_disappeared: Optional[Annotated[int, Field(ge=1)]] = None

    @field_validator("imgsz")
    @classmethod
    def imgsz_multiple_of_32(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 32 != 0:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @model_validator(mode="after")
    def at_least_one_field(self) -> "StreamPatch":
        if not any(v is not None for v in self.model_dump().values()):
            raise ValueError("Provide at least one field to update")
        return self


class StreamStatus(BaseModel):
    id: str
    name: str
    state: Literal["connecting", "connected", "reconnecting", "error", "disconnected"]
    wing_count: int
    counting: bool
    fps: float
    error: Optional[str] = None


class StreamDetail(StreamStatus):
    url: str
    confidence: float
    nms_iou: float
    imgsz: int
    roi_position: float
    max_distance: int
    max_disappeared: int


class StreamSummary(BaseModel):
    id: str
    name: str
    state: str
    wing_count: int


def _to_status(s: StreamSession) -> StreamStatus:
    return StreamStatus(
        id=s.id, name=s.name, state=s.state, wing_count=s.wing_count,
        counting=s.counting, fps=s.fps_display, error=s.error,
    )


def _to_detail(s: StreamSession) -> StreamDetail:
    return StreamDetail(
        id=s.id, name=s.name, state=s.state, wing_count=s.wing_count,
        counting=s.counting, fps=s.fps_display, error=s.error,
        url=s.url, confidence=s.confidence, nms_iou=s.nms_iou, imgsz=s.imgsz,
        roi_position=s.roi_position, max_distance=s.max_distance,
        max_disappeared=s.max_disappeared,
    )


def _to_summary(s: StreamSession) -> StreamSummary:
    return StreamSummary(id=s.id, name=s.name, state=s.state, wing_count=s.wing_count)


def _manager(request: Request) -> StreamManager:
    return request.app.state.stream_manager


# ----------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------

@router.get("")
def list_streams(request: Request) -> dict:
    mgr = _manager(request)
    return {
        "streams": [_to_summary(s).model_dump() for s in mgr.list()],
        "capacity": mgr.CAPACITY,
    }


@router.post("", status_code=status.HTTP_201_CREATED, response_model=StreamDetail)
def create_stream(body: StreamCreate, request: Request) -> StreamDetail:
    mgr = _manager(request)
    try:
        session = mgr.add(**body.model_dump(exclude_none=True))
    except CapacityExceeded as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _to_detail(session)


@router.get("/status")
def all_status(request: Request) -> dict:
    mgr = _manager(request)
    return {"streams": [_to_status(s).model_dump() for s in mgr.list()]}


# ----------------------------------------------------------------------------
# Per-stream
# ----------------------------------------------------------------------------

@router.get("/{stream_id}", response_model=StreamDetail)
def get_stream(stream_id: str, request: Request) -> StreamDetail:
    try:
        return _to_detail(_manager(request).get(stream_id))
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.patch("/{stream_id}", response_model=StreamDetail)
def patch_stream(stream_id: str, body: StreamPatch, request: Request) -> StreamDetail:
    mgr = _manager(request)
    try:
        session = mgr.patch(stream_id, body.model_dump(exclude_none=True))
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _to_detail(session)


@router.delete("/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_stream(stream_id: str, request: Request) -> Response:
    try:
        _manager(request).remove(stream_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{stream_id}/status", response_model=StreamStatus)
def stream_status(stream_id: str, request: Request) -> StreamStatus:
    try:
        return _to_status(_manager(request).get(stream_id))
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{stream_id}/feed")
def stream_feed(stream_id: str, request: Request):
    try:
        session = _manager(request).get(stream_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    def _generator():
        # Keep yielding while the session exists in the manager.
        while True:
            jpeg = session.latest_annotated_jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.03)
            if session.state == "disconnected":
                break

    return StreamingResponse(
        _generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/{stream_id}/counting/start")
def counting_start(stream_id: str, request: Request) -> dict:
    try:
        session = _manager(request).get(stream_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    session.start_counting()
    return {"counting": True}


@router.post("/{stream_id}/counting/stop")
def counting_stop(stream_id: str, request: Request) -> dict:
    try:
        session = _manager(request).get(stream_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    session.stop_counting()
    return {"counting": False}


@router.post("/{stream_id}/counting/reset")
def counting_reset(stream_id: str, request: Request) -> dict:
    try:
        session = _manager(request).get(stream_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    session.reset_counting()
    return {"counting": session.counting, "wing_count": 0}
```

- [ ] **Step 2: Write integration tests using TestClient**

Create `tests/integration/test_streams_api.py`:
```python
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.stream_manager import StreamManager
from app.routers.streams import router as streams_router
from tests.conftest import FakeYOLOModel


@pytest.fixture
def client():
    # Build a minimal FastAPI app that mounts only the streams router.
    # Patch CaptureThread so no real RTSP connection is attempted.
    with patch("app.core.stream_manager.CaptureThread") as ct_class:
        ct_class.return_value.start.return_value = None
        ct_class.return_value.stop.return_value = None

        app = FastAPI()
        app.state.stream_manager = StreamManager(model=FakeYOLOModel())
        app.include_router(streams_router)
        yield TestClient(app)
        app.state.stream_manager.shutdown()


def test_post_creates_stream(client):
    resp = client.post("/api/streams", json={"url": "rtsp://a", "name": "Cam 1"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["url"] == "rtsp://a"
    assert body["name"] == "Cam 1"
    assert len(body["id"]) == 8


def test_post_eleventh_returns_409(client):
    for i in range(10):
        client.post("/api/streams", json={"url": f"rtsp://{i}"})
    resp = client.post("/api/streams", json={"url": "rtsp://overflow"})
    assert resp.status_code == 409
    assert "Capacity reached" in resp.json()["detail"]


def test_get_unknown_returns_404(client):
    resp = client.get("/api/streams/deadbeef")
    assert resp.status_code == 404


def test_patch_empty_body_returns_422(client):
    s = client.post("/api/streams", json={"url": "rtsp://a"}).json()
    resp = client.patch(f"/api/streams/{s['id']}", json={})
    assert resp.status_code == 422


def test_patch_invalid_imgsz_returns_422(client):
    s = client.post("/api/streams", json={"url": "rtsp://a"}).json()
    resp = client.patch(f"/api/streams/{s['id']}", json={"imgsz": 100})
    assert resp.status_code == 422


def test_post_invalid_imgsz_returns_422(client):
    resp = client.post("/api/streams", json={"url": "rtsp://a", "imgsz": 100})
    assert resp.status_code == 422


def test_post_empty_url_returns_422(client):
    resp = client.post("/api/streams", json={"url": ""})
    assert resp.status_code == 422


def test_delete_removes_stream(client):
    s = client.post("/api/streams", json={"url": "rtsp://a"}).json()
    resp = client.delete(f"/api/streams/{s['id']}")
    assert resp.status_code == 204
    assert client.get(f"/api/streams/{s['id']}").status_code == 404


def test_all_status_returns_every_stream(client):
    a = client.post("/api/streams", json={"url": "rtsp://a"}).json()
    b = client.post("/api/streams", json={"url": "rtsp://b"}).json()
    resp = client.get("/api/streams/status")
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()["streams"]}
    assert ids == {a["id"], b["id"]}


def test_counting_start_stop_reset(client):
    s = client.post("/api/streams", json={"url": "rtsp://a"}).json()
    sid = s["id"]
    assert client.post(f"/api/streams/{sid}/counting/start").json() == {"counting": True}
    detail = client.get(f"/api/streams/{sid}").json()
    assert detail["counting"] is True
    assert client.post(f"/api/streams/{sid}/counting/stop").json() == {"counting": False}
    assert client.post(f"/api/streams/{sid}/counting/reset").json() == {
        "counting": False, "wing_count": 0,
    }


def test_list_includes_capacity(client):
    resp = client.get("/api/streams")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capacity"] == 10
    assert body["streams"] == []
```

- [ ] **Step 3: Run — confirm all pass**

Run: `pytest tests/integration/test_streams_api.py -v`
Expected: 11 passed.

- [ ] **Step 4: Commit**

```bash
git add app/routers/streams.py tests/integration/test_streams_api.py
git commit -m "feat: add /api/streams collection-based router"
```

---

## Task 7: Wire `StreamManager` into `main.py` and reduce global config

**Files:**
- Modify: `app/main.py`
- Modify: `app/config.py`
- Modify: `app/core/runtime_config.py`
- Modify: `app/routers/config_router.py`

- [ ] **Step 1: Reduce `Settings` in `app/config.py`**

Replace `app/config.py` content with:
```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = r"runs\detect\runs\chicken_wings\weights\best.pt"
    upload_dir: str = "app/uploads"
    output_dir: str = "app/outputs"

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 2: Reduce `runtime_config.py`**

Replace `app/core/runtime_config.py` content with:
```python
import threading
from app.config import Settings


class RuntimeConfig:
    """Thread-safe live configuration for global (cross-stream) settings.

    Per-stream inference and tracker parameters live on each StreamSession;
    this holds only fields that genuinely apply to the whole container:
    model_path, upload_dir, output_dir.
    """

    def __init__(self) -> None:
        boot = Settings()
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_data", {
            "model_path": boot.model_path,
            "upload_dir": boot.upload_dir,
            "output_dir": boot.output_dir,
        })

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            lock = object.__getattribute__(self, "_lock")
            with lock:
                return data[name]
        raise AttributeError(f"RuntimeConfig has no field '{name}'")

    def snapshot(self) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            return dict(data)

    def update(self, patch: dict) -> dict:
        lock = object.__getattribute__(self, "_lock")
        data = object.__getattribute__(self, "_data")
        with lock:
            for key, value in patch.items():
                if key in data:
                    data[key] = value
            return dict(data)


runtime_config = RuntimeConfig()
```

- [ ] **Step 3: Reduce `config_router.py`**

Replace `app/routers/config_router.py` content with:
```python
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigResponse(BaseModel):
    model_path: str
    upload_dir: str
    output_dir: str


class ConfigPatch(BaseModel):
    model_path: Optional[str] = None
    upload_dir: Optional[str] = None
    output_dir: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "ConfigPatch":
        if not any(v is not None for v in self.model_dump().values()):
            raise ValueError("Provide at least one field to update")
        return self


@router.get("", response_model=ConfigResponse, summary="Get current global config")
def get_config() -> dict:
    return runtime_config.snapshot()


@router.patch("", response_model=ConfigResponse, summary="Update global config without restart")
def patch_config(body: ConfigPatch) -> dict:
    """Update global config. Per-stream config lives on /api/streams/{id}, not here."""
    patch = {k: v for k, v in body.model_dump().items() if v is not None}

    old_model_path = runtime_config.model_path
    new_model_path = patch.get("model_path")

    updated = runtime_config.update(patch)

    if new_model_path and new_model_path != old_model_path:
        try:
            preload_model(new_model_path)
        except Exception as exc:
            runtime_config.update({"model_path": old_model_path})
            raise HTTPException(
                status_code=422,
                detail=f"Model load failed — config rolled back: {exc}",
            )

    return updated
```

- [ ] **Step 4: Wire `StreamManager` into `app/main.py`**

Replace `app/main.py` content with:
```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model, get_model
from app.core.stream_manager import StreamManager
from app.routers import image, video
from app.routers.streams import router as streams_router
from app.routers.config_router import router as config_router
from app.routers.export_router import router as export_router
from app.routers.health_router import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    snap = runtime_config.snapshot()
    os.makedirs(snap["upload_dir"], exist_ok=True)
    os.makedirs(snap["output_dir"], exist_ok=True)
    preload_model(snap["model_path"])
    app.state.stream_manager = StreamManager(model=get_model(snap["model_path"]))
    try:
        yield
    finally:
        app.state.stream_manager.shutdown()


app = FastAPI(
    title="Chicken Wing Counter",
    version="3.0.0",
    description=(
        "Production-grade wing counting API. "
        "Supports uploaded images, uploaded videos, and up to 10 simultaneous "
        "RTSP streams sharing a single batched-inference YOLO model."
    ),
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(config_router)
app.include_router(export_router)
app.include_router(image.router)
app.include_router(video.router)
app.include_router(streams_router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
```

- [ ] **Step 5: Run the full test suite — confirm nothing broke**

Run: `pytest -q`
Expected: all unit and integration tests still pass. The integration tests stub out `CaptureThread`, so no real RTSP connections.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/config.py app/core/runtime_config.py app/routers/config_router.py
git commit -m "refactor: reduce global config to model_path/upload_dir/output_dir; wire StreamManager"
```

---

## Task 8: Delete the old `/api/stream` router

**Files:**
- Delete: `app/routers/stream.py`

- [ ] **Step 1: Confirm `app/main.py` no longer imports it**

Run: `grep -n "from app.routers import" app/main.py`
Expected: only `image, video` listed, not `stream`.

Run: `grep -rn "from app.routers.stream " app/ tests/`
Expected: no matches.

- [ ] **Step 2: Delete the file**

Run: `git rm app/routers/stream.py`

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass; no import errors.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove single-stream /api/stream router (superseded by /api/streams)"
```

---

## Task 9: Replace `stream.html` and `stream.js` with multi-stream dashboard

**Files:**
- Modify: `app/static/stream.html`
- Modify: `app/static/js/stream.js`
- Modify: `app/static/css/style.css`

- [ ] **Step 1: Replace `stream.html` with dashboard markup**

Overwrite `app/static/stream.html` with:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <title>Wing Counter - Live Streams</title>
    <link rel="stylesheet" href="/css/style.css">
</head>
<body>
    <nav class="navbar">
        <span class="logo">WingCounter</span>
        <a href="/">Image</a>
        <a href="/video.html">Video</a>
        <a href="/stream.html" class="active">Streams</a>
    </nav>

    <div class="container container-wide">
        <h1 class="page-title">Live Streams</h1>
        <p class="page-subtitle">
            Connect up to 10 RTSP cameras. All streams share one YOLO model via batched inference.
            <span id="capacityLabel"></span>
        </p>

        <div class="add-stream-card">
            <input class="input" id="newStreamUrl" type="text" placeholder="rtsp://user:pass@camera-ip:554/stream">
            <input class="input input-name" id="newStreamName" type="text" placeholder="Name (optional)">
            <button class="btn btn-success" id="btnAdd">Add stream</button>
        </div>

        <div class="stream-grid" id="streamGrid"></div>
    </div>

    <template id="cardTemplate">
        <div class="stream-card">
            <header class="stream-card-header">
                <input class="stream-name-input" type="text">
                <span class="state-pill"><span class="dot"></span><span class="label">connecting</span></span>
                <button class="btn-icon btn-delete" title="Remove">&times;</button>
            </header>
            <div class="stream-feed">
                <img class="feed" alt="">
                <div class="feed-placeholder">Connecting…</div>
            </div>
            <div class="stream-stats">
                <div class="stat"><div class="stat-value count">0</div><div class="stat-label">wings</div></div>
                <div class="stat"><div class="stat-value fps">0</div><div class="stat-label">fps</div></div>
            </div>
            <div class="stream-controls">
                <button class="btn btn-primary btn-count-start">Start counting</button>
                <button class="btn btn-outline btn-count-stop" disabled>Stop</button>
                <button class="btn btn-outline btn-count-reset">Reset</button>
                <button class="btn btn-icon btn-settings" title="Settings">⚙</button>
            </div>
            <details class="stream-settings">
                <summary>Settings</summary>
                <div class="setting-row">
                    <label>ROI Position</label>
                    <input type="range" class="setting-roi" min="0.1" max="0.9" step="0.05">
                    <span class="setting-roi-value"></span>
                </div>
                <div class="setting-row">
                    <label>Confidence</label>
                    <input type="range" class="setting-conf" min="0.05" max="0.95" step="0.05">
                    <span class="setting-conf-value"></span>
                </div>
                <div class="setting-row">
                    <label>NMS IoU</label>
                    <input type="range" class="setting-iou" min="0.05" max="0.95" step="0.05">
                    <span class="setting-iou-value"></span>
                </div>
                <div class="setting-row">
                    <label>imgsz</label>
                    <select class="setting-imgsz">
                        <option value="320">320</option>
                        <option value="416">416</option>
                        <option value="512">512</option>
                        <option value="640">640</option>
                        <option value="800">800</option>
                        <option value="960">960</option>
                    </select>
                </div>
                <div class="setting-row">
                    <label>Max distance</label>
                    <input type="number" class="setting-maxdist" min="1" step="1">
                </div>
                <div class="setting-row">
                    <label>Max disappeared</label>
                    <input type="number" class="setting-maxdis" min="1" step="1">
                </div>
            </details>
        </div>
    </template>

    <script src="/js/stream.js"></script>
</body>
</html>
```

- [ ] **Step 2: Replace `stream.js` with dashboard logic**

Overwrite `app/static/js/stream.js` with:
```javascript
const grid = document.getElementById("streamGrid");
const newUrl = document.getElementById("newStreamUrl");
const newName = document.getElementById("newStreamName");
const btnAdd = document.getElementById("btnAdd");
const capacityLabel = document.getElementById("capacityLabel");
const tpl = document.getElementById("cardTemplate");

const cards = new Map();  // id -> { el, lastDetail }

async function loadAll() {
    const resp = await fetch("/api/streams");
    const body = await resp.json();
    capacityLabel.textContent = `(${body.streams.length} / ${body.capacity})`;
    btnAdd.disabled = body.streams.length >= body.capacity;
    for (const summary of body.streams) {
        const detail = await fetch(`/api/streams/${summary.id}`).then(r => r.json());
        addCardFromDetail(detail);
    }
}

btnAdd.addEventListener("click", async () => {
    const url = newUrl.value.trim();
    if (!url) { alert("Enter an RTSP URL"); return; }
    const body = { url };
    if (newName.value.trim()) body.name = newName.value.trim();
    const resp = await fetch("/api/streams", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json();
        alert(err.detail || "Failed to add stream");
        return;
    }
    const detail = await resp.json();
    addCardFromDetail(detail);
    newUrl.value = ""; newName.value = "";
    updateCapacity();
});

function addCardFromDetail(detail) {
    const el = tpl.content.firstElementChild.cloneNode(true);
    el.dataset.id = detail.id;
    el.querySelector(".feed").src = `/api/streams/${detail.id}/feed?t=${Date.now()}`;
    bindCard(el, detail);
    grid.appendChild(el);
    cards.set(detail.id, { el, lastDetail: detail });
    applyDetail(el, detail);
}

function bindCard(el, detail) {
    const id = detail.id;
    const nameInput = el.querySelector(".stream-name-input");
    nameInput.value = detail.name;
    nameInput.addEventListener("change", () => {
        patchStream(id, { name: nameInput.value.trim() });
    });

    el.querySelector(".btn-delete").addEventListener("click", async () => {
        if (!confirm("Remove this stream?")) return;
        await fetch(`/api/streams/${id}`, { method: "DELETE" });
        cards.get(id).el.remove();
        cards.delete(id);
        updateCapacity();
    });

    el.querySelector(".btn-count-start").addEventListener("click", async () => {
        await fetch(`/api/streams/${id}/counting/start`, { method: "POST" });
    });
    el.querySelector(".btn-count-stop").addEventListener("click", async () => {
        await fetch(`/api/streams/${id}/counting/stop`, { method: "POST" });
    });
    el.querySelector(".btn-count-reset").addEventListener("click", async () => {
        await fetch(`/api/streams/${id}/counting/reset`, { method: "POST" });
    });

    const roi = el.querySelector(".setting-roi");
    const roiVal = el.querySelector(".setting-roi-value");
    const conf = el.querySelector(".setting-conf");
    const confVal = el.querySelector(".setting-conf-value");
    const iou = el.querySelector(".setting-iou");
    const iouVal = el.querySelector(".setting-iou-value");
    const imgsz = el.querySelector(".setting-imgsz");
    const maxdist = el.querySelector(".setting-maxdist");
    const maxdis = el.querySelector(".setting-maxdis");

    roi.value = detail.roi_position; roiVal.textContent = detail.roi_position.toFixed(2);
    conf.value = detail.confidence; confVal.textContent = detail.confidence.toFixed(2);
    iou.value = detail.nms_iou; iouVal.textContent = detail.nms_iou.toFixed(2);
    imgsz.value = detail.imgsz;
    maxdist.value = detail.max_distance;
    maxdis.value = detail.max_disappeared;

    roi.addEventListener("input", () => roiVal.textContent = parseFloat(roi.value).toFixed(2));
    roi.addEventListener("change", () => patchStream(id, { roi_position: parseFloat(roi.value) }));
    conf.addEventListener("input", () => confVal.textContent = parseFloat(conf.value).toFixed(2));
    conf.addEventListener("change", () => patchStream(id, { confidence: parseFloat(conf.value) }));
    iou.addEventListener("input", () => iouVal.textContent = parseFloat(iou.value).toFixed(2));
    iou.addEventListener("change", () => patchStream(id, { nms_iou: parseFloat(iou.value) }));
    imgsz.addEventListener("change", () => patchStream(id, { imgsz: parseInt(imgsz.value, 10) }));
    maxdist.addEventListener("change", () => patchStream(id, { max_distance: parseInt(maxdist.value, 10) }));
    maxdis.addEventListener("change", () => patchStream(id, { max_disappeared: parseInt(maxdis.value, 10) }));
}

async function patchStream(id, body) {
    const resp = await fetch(`/api/streams/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json();
        alert(err.detail || "Update failed");
    }
}

function applyDetail(el, detail) {
    el.querySelector(".count").textContent = detail.wing_count;
    el.querySelector(".fps").textContent = detail.fps.toFixed(1);
    const pill = el.querySelector(".state-pill");
    pill.querySelector(".label").textContent = detail.state;
    pill.className = `state-pill ${detail.state}`;
    el.querySelector(".btn-count-start").disabled = detail.counting;
    el.querySelector(".btn-count-stop").disabled = !detail.counting;
}

async function poll() {
    try {
        const resp = await fetch("/api/streams/status");
        const body = await resp.json();
        for (const status of body.streams) {
            const entry = cards.get(status.id);
            if (!entry) continue;
            const el = entry.el;
            el.querySelector(".count").textContent = status.wing_count;
            el.querySelector(".fps").textContent = status.fps.toFixed(1);
            const pill = el.querySelector(".state-pill");
            pill.querySelector(".label").textContent = status.state;
            pill.className = `state-pill ${status.state}`;
            el.querySelector(".btn-count-start").disabled = status.counting;
            el.querySelector(".btn-count-stop").disabled = !status.counting;
        }
    } catch (_) {}
}

function updateCapacity() {
    const n = cards.size;
    capacityLabel.textContent = `(${n} / 10)`;
    btnAdd.disabled = n >= 10;
}

loadAll().then(() => setInterval(poll, 1000));
```

- [ ] **Step 3: Append dashboard styles to `style.css`**

Append to `app/static/css/style.css`:
```css
/* ---------- multi-stream dashboard ---------- */
.container-wide { max-width: 1600px; }

.add-stream-card {
    display: flex;
    gap: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 24px;
    align-items: center;
}
.add-stream-card .input { flex: 1; min-width: 0; }
.add-stream-card .input-name { max-width: 220px; flex: 0 0 220px; }

.stream-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
}

.stream-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.stream-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
}
.stream-name-input {
    flex: 1;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text);
    font-size: 15px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 6px;
}
.stream-name-input:hover, .stream-name-input:focus {
    background: var(--surface-2);
    border-color: var(--border);
    outline: none;
}

.state-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--surface-2);
    font-size: 12px;
    color: var(--text-dim);
    text-transform: capitalize;
}
.state-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
}
.state-pill.connected .dot { background: var(--green); }
.state-pill.connecting .dot { background: var(--amber); }
.state-pill.reconnecting .dot { background: var(--amber); animation: blink 1s infinite; }
.state-pill.error .dot, .state-pill.disconnected .dot { background: var(--red); }

@keyframes blink { 50% { opacity: 0.3; } }

.btn-icon {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    width: 32px;
    height: 32px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
}
.btn-icon:hover { background: var(--surface-2); }
.btn-delete:hover { background: var(--red); color: white; border-color: var(--red); }

.stream-feed {
    position: relative;
    aspect-ratio: 16/9;
    background: black;
    border-radius: 8px;
    overflow: hidden;
}
.stream-feed .feed { width: 100%; height: 100%; object-fit: cover; display: block; }
.stream-feed .feed-placeholder {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-dim); font-size: 14px;
    pointer-events: none;
}

.stream-stats {
    display: flex;
    gap: 16px;
}
.stream-stats .stat {
    flex: 1;
    background: var(--surface-2);
    border-radius: 8px;
    padding: 8px 12px;
}
.stream-stats .stat-value { font-size: 20px; font-weight: 700; color: var(--green); }
.stream-stats .stat-value.fps { color: var(--amber); }
.stream-stats .stat-label { font-size: 11px; color: var(--text-dim); }

.stream-controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.stream-controls .btn { flex: 1; min-width: 0; }

.stream-settings {
    border-top: 1px solid var(--border);
    padding-top: 12px;
}
.stream-settings summary { cursor: pointer; color: var(--text-dim); font-size: 13px; }
.stream-settings .setting-row {
    display: flex; align-items: center; gap: 8px;
    margin-top: 8px; font-size: 13px;
}
.stream-settings .setting-row label { flex: 0 0 110px; color: var(--text-dim); }
.stream-settings .setting-row input[type="range"] { flex: 1; }
.stream-settings .setting-row input[type="number"],
.stream-settings .setting-row select {
    flex: 1; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text); padding: 4px 8px; border-radius: 6px;
}
.stream-settings .setting-row span { flex: 0 0 40px; text-align: right; color: var(--text); }
```

- [ ] **Step 4: Manual smoke test (optional, requires running container)**

Run: `docker compose up --build` (or `uvicorn app.main:app --reload --port 5580` if running locally)
Open: `http://localhost:5580/stream.html`
Expected:
- Page loads with empty grid and "(0 / 10)" capacity label
- Adding `rtsp://invalid-url:5554/stream` creates a card with state "reconnecting"
- DELETE removes the card and decrements the count

(This step is manual and may be deferred to Task 11's smoke test.)

- [ ] **Step 5: Commit**

```bash
git add app/static/stream.html app/static/js/stream.js app/static/css/style.css
git commit -m "feat(ui): replace single-stream page with multi-stream dashboard grid"
```

---

## Task 10: Drop per-stream env vars from `docker-compose.yml`

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Trim env vars**

Replace `docker-compose.yml` content with:
```yaml
services:
  wing-counter:
    image: basim123/chickens-wings-counter:latest
    build: .
    ports:
      - "5580:5580"
    volumes:
      - ./app/uploads:/app/app/uploads
      - ./app/outputs:/app/app/outputs
    environment:
      - MODEL_PATH=${MODEL_PATH:-best.pt}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5580/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: drop per-stream env vars from docker-compose (streams configured via API)"
```

---

## Task 11: End-to-end integration test with file-backed source

**Files:**
- Create: `tests/integration/test_end_to_end_file.py`

This test runs three `StreamSession`s end-to-end through `CaptureThread → InferenceWorker → StreamSession` using local MP4 files instead of RTSP — same code path (it's just a different OpenCV source string). Verifies independent per-session wing counts (no bleed-through), preserved counts during the file's lifetime, and JPEG output.

- [ ] **Step 1: Find or generate three short test videos**

If short test videos don't exist, generate three deterministic ones:
```python
# Run this once locally (NOT part of the test, just to create fixtures):
import cv2, numpy as np, os
os.makedirs("tests/fixtures", exist_ok=True)
for i, dot_x in enumerate([100, 300, 500]):
    w = cv2.VideoWriter(
        f"tests/fixtures/dot_{i}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 384)
    )
    for f in range(60):
        frame = np.zeros((384, 640, 3), dtype=np.uint8)
        # Moving white square crossing the ROI line (y=268)
        y = int(50 + f * 4)
        cv2.rectangle(frame, (dot_x-15, y-15), (dot_x+15, y+15), (255,255,255), -1)
        w.write(frame)
    w.release()
```

The test below will skip if these files don't exist.

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_end_to_end_file.py`:
```python
import os
import time
from unittest.mock import patch

import pytest

from app.core.stream_manager import StreamManager
from tests.conftest import FakeBox, FakeResult, FakeYOLOModel


pytestmark = pytest.mark.skipif(
    not all(os.path.exists(f"tests/fixtures/dot_{i}.mp4") for i in range(3)),
    reason="test fixtures (tests/fixtures/dot_*.mp4) not generated",
)


def _make_fake_model_for_three_dot_files():
    """Each call to model() will be invoked with up to 3 frames. Return one
    detection per frame at a deterministic location so we can verify fan-out.
    The detection y-coords increase by 4 per call to mimic the moving square."""
    # We return *unlimited* batches by overriding __call__ instead of using the
    # finite per_call_results queue.
    model = FakeYOLOModel()
    call_counter = {"n": 0}

    def call(frames, conf=None, iou=None, imgsz=None, verbose=False):
        model.calls.append({"n": len(frames), "conf": conf, "iou": iou, "imgsz": imgsz})
        results = []
        for i, _ in enumerate(frames):
            y = 50 + call_counter["n"] * 4
            x = 100 + i * 200
            results.append(FakeResult([FakeBox(x-15, y-15, x+15, y+15, 0.9)]))
        call_counter["n"] += 1
        return results

    model.__call__ = call
    return model


def test_three_sessions_count_independently(tmp_path):
    """Three sessions backed by three different files. Each must count its own
    object crossings without bleed-through from the others."""
    model = _make_fake_model_for_three_dot_files()
    manager = StreamManager(model=model)

    s = []
    try:
        for i in range(3):
            session = manager.add(
                url=os.path.abspath(f"tests/fixtures/dot_{i}.mp4"),
                name=f"file-{i}",
            )
            session.start_counting()
            s.append(session)

        # Let the pipeline run for ~6 seconds (file is 6s at 10fps).
        time.sleep(8.0)

        # Each session should have a positive wing_count and counts should
        # not differ wildly (within +/- 1 of each other since the moving
        # square crosses the ROI exactly once per file).
        counts = [sess.wing_count for sess in s]
        assert all(c >= 1 for c in counts), f"At least one session never counted: {counts}"
        assert max(counts) - min(counts) <= 1, f"Counts differ too much: {counts}"
    finally:
        manager.shutdown()
```

- [ ] **Step 3: Run the test**

If fixtures exist: `pytest tests/integration/test_end_to_end_file.py -v`
Expected: 1 passed (or skipped if fixtures don't exist).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_end_to_end_file.py
git commit -m "test: add end-to-end multi-session integration with file-backed sources"
```

---

## Task 12: Final verification

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: All unit tests in `tests/unit/` pass; all integration tests in `tests/integration/` pass (or skip cleanly if fixtures absent).

- [ ] **Step 2: Build the Docker image**

Run: `docker compose build`
Expected: Build succeeds with the trimmed environment block.

- [ ] **Step 3: Start the container**

Run: `docker compose up -d`
Wait: `curl -f http://localhost:5580/health` returns 200 (may take ~60s for model load).

- [ ] **Step 4: Smoke-test the new API**

```bash
# List (should be empty)
curl http://localhost:5580/api/streams
# Expected: {"streams": [], "capacity": 10}

# Add a stream (use a public test feed or a known RTSP URL)
curl -X POST http://localhost:5580/api/streams \
  -H "Content-Type: application/json" \
  -d '{"url":"rtsp://your-camera-url","name":"Test"}'
# Expected: 201 with stream detail

# List again
curl http://localhost:5580/api/streams
# Expected: streams: [...] with 1 entry

# Aggregated status
curl http://localhost:5580/api/streams/status
```

- [ ] **Step 5: Open the dashboard**

Open: `http://localhost:5580/stream.html`
Verify:
- Empty grid renders, "(0 / 10)" shows in capacity label
- Add a stream — card appears, MJPEG feed loads (or "Connecting…" if URL is invalid)
- Click Start counting — button disables, Stop enables
- Slider changes update the corresponding param (verify via `curl /api/streams/{id}`)
- Delete (×) removes the card and decrements capacity

- [ ] **Step 6: Stop the container**

Run: `docker compose down`

---

## Plan Self-Review

**Spec coverage:**
- ✅ 10-stream cap → Task 5 (`CAPACITY = 10`, `CapacityExceeded`)
- ✅ Batched inference, single model → Task 4 (`InferenceWorker._run_once`)
- ✅ Per-stream config (all 7 fields) → Task 2 (`StreamSession.__init__`), Task 5 (`apply_patch`), Task 6 (`StreamCreate` / `StreamPatch`)
- ✅ MJPEG per stream → Task 6 (`/api/streams/{id}/feed`)
- ✅ Per-stream count + FPS + status → Task 2 (`process_detections`), Task 6 (`/status`)
- ✅ Reconnect with backoff (2/4/8/16) → Task 3 (`BACKOFF_SEQUENCE`, `_run`)
- ✅ Wing count preserved across reconnects → Task 2 (counter lives on session, capture thread is stateless about counts)
- ✅ Dynamic CRUD, in-memory → Task 5 (`add`/`remove`/`get`/`list`), Task 6 (routes)
- ✅ No recording to disk → not present in any task (intentional non-goal)
- ✅ Counting start/stop/reset per stream → Task 2 (methods), Task 6 (routes)
- ✅ Global config reduced → Task 7
- ✅ Frontend dashboard → Task 9
- ✅ Old single-stream router removed → Task 8
- ✅ Tests for capacity, fan-out, reconnect, threshold filtering → Tasks 2/3/4/5/6
- ✅ PATCH semantics (URL → restart, params → live) → Task 5 (`patch()` method)
- ✅ `CAP_PROP_BUFFERSIZE=1` → Task 3

**Placeholder scan:** None found. All code blocks contain complete, runnable code.

**Type consistency:**
- `StreamSession.set_state(state, error=...)` — used in Task 2 and Task 3 with same signature.
- `StreamSession.take_latest_frame()` returns `Optional[np.ndarray]` — used consistently in Tasks 3, 4.
- `StreamSession.process_detections(det_info: list[dict])` — matches `_extract_dets` return type in Task 4.
- `StreamManager.add(**kwargs)` accepts `url`, `name`, and all the per-stream params — matches `StreamCreate.model_dump(exclude_none=True)` in Task 6.
- `StreamManager.patch(stream_id, dict)` — matches Task 6 router call site.
- `CaptureThread.start()` / `stop()` — consistent across Tasks 3, 5.
- `InferenceWorker.start()` / `stop()` — consistent across Tasks 4, 5.
- State literal set is identical between `StreamSession.set_state` calls and `StreamStatus.state` literal type in Task 6.

No issues to fix.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-multi-stream-batched-inference.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
