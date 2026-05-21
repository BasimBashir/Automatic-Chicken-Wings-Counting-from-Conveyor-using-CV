# Multi-Stream RTSP Support with Batched Inference

**Date:** 2026-05-21
**Status:** Approved
**Scope:** Replace the single-stream `/api/stream/*` endpoint with a multi-stream collection API supporting up to 10 simultaneous RTSP streams, each with independent configuration. A single YOLO model runs batched inference across all connected streams.

---

## Goals

- Support up to **10 simultaneous RTSP streams** per container.
- **Batched inference**: a single shared YOLO model serves all streams in one `model(batch, ...)` call per cycle, maximizing GPU utilization.
- **Per-stream results**: each stream returns its own MJPEG feed, wing count, FPS, and status, isolated from the others.
- **Per-stream configuration**: confidence, NMS IoU, imgsz, ROI position, tracker `max_distance` and `max_disappeared` are all configurable per stream at runtime via PATCH.
- **Resilient capture**: on RTSP disconnect, auto-reconnect with exponential backoff (2 → 4 → 8 → 16s, then cap at 16s) indefinitely. Wing count is preserved across reconnects.
- **Single container** deployment on a VPS — no orchestration changes beyond what's already in `docker-compose.yml`.

## Non-goals

- Persistence of stream configs across container restarts (v1 is in-memory only).
- Recording annotated streams to disk (RTSP streams are intended to run 24/7; recording would fill the VPS quickly). The existing upload-video flow continues to record.
- WebSocket / push-based count updates (clients poll `/status`).
- Authentication on the API (out of scope; deploy behind reverse proxy with auth if needed).
- Per-stream model selection — all streams share the model configured globally via `MODEL_PATH`.

---

## Architecture

```
┌──────────────────┐   latest frame   ┌─────────────────────┐
│ CaptureThread 0  │ ───────────────▶ │                     │
│  (RTSP read)     │                  │                     │
├──────────────────┤                  │  InferenceWorker    │
│ CaptureThread 1  │ ───────────────▶ │  (single thread)    │   batched
├──────────────────┤   single-slot    │                     │   model() call
│      ...         │   overwrite      │  Every ~30ms:       │ ─────────────▶ YOLO
├──────────────────┤                  │   1. snapshot slots │
│ CaptureThread N  │ ───────────────▶ │   2. batch infer    │
└──────────────────┘                  │   3. fan out result │
                                      └──────────┬──────────┘
                                                 │ per-stream det_info
                                                 ▼
                              ┌────────────────────────────────────┐
                              │ StreamSession 0..N                 │
                              │  - WingCounter (own ROI, params)   │
                              │  - annotator                       │
                              │  - latest_frame (JPEG bytes)       │
                              │  - status (count, fps, state)      │
                              └────────────────────────────────────┘
```

### Components

**`StreamSession`** (`app/core/stream_session.py`)
One per stream. Owns:
- `id` (8-char hex), `name` (human label), `url` (RTSP)
- Per-stream config: `confidence`, `nms_iou`, `imgsz`, `roi_position`, `max_distance`, `max_disappeared`
- A `WingCounter` instance (from `app/core/counter.py`, unchanged)
- Single-slot capture frame buffer (overwritten by capture thread, drained by inference worker)
- Latest annotated JPEG bytes (consumed by the MJPEG feed endpoint)
- Status fields: `state` (`connecting` / `connected` / `reconnecting` / `error` / `disconnected`), `counting` (bool), `wing_count`, `fps_display`, `error`
- `RLock` guarding all mutable state

**`CaptureThread`** (`app/core/capture_thread.py`)
One per `StreamSession`. Opens `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`, sets `CAP_PROP_BUFFERSIZE=1`, reads frames in a loop, and writes only the **latest** frame into the session's slot (drop-oldest). On read failure, closes the capture and reconnects with exponential backoff. The backoff sleep is interruptible (returns immediately on `stop()`).

**`InferenceWorker`** (`app/core/inference_worker.py`)
Single global thread, started when the first stream is added and stopped when the last stream is removed. Each cycle:
1. Snapshot every `connected` session's latest frame (`session.take_latest_frame()` returns-and-clears).
2. If any frames present, call `model(batch, conf=0.01, iou=0.99, imgsz=max_imgsz, verbose=False)` once. The permissive thresholds ensure all candidate boxes come back; per-stream `confidence` and `nms_iou` are applied post-inference.
3. Fan out: for each `(session, result)` pair, extract `det_info` filtered by that session's thresholds and call `session.process_detections(det_info)`.
4. Sleep the remainder of the 30ms batch window. If inference took >30ms, the next cycle starts immediately.

**`StreamManager`** (`app/core/stream_manager.py`)
Registry of `dict[str, StreamSession]`. Owns add/remove/list, enforces the 10-stream cap, owns the `InferenceWorker` lifecycle. Created in the FastAPI lifespan and stored on `app.state`.

### Concurrency model

- Each session has its own `RLock` guarding its mutable state.
- The capture thread is the **single writer** of the session's frame slot — no contention with itself.
- The inference worker is the **single reader** of all frame slots and the **single writer** of `det_info → counter → annotated_frame`. Counter updates are therefore implicitly serialized per session; the counter does not need its own lock.
- The MJPEG feed endpoint reads `latest_annotated_jpeg` under the session lock.
- The `StreamManager` has its own lock for the sessions dict.

### Why post-inference filtering for per-stream conf/iou

Ultralytics' `model(batch, conf=X, iou=Y)` applies single thresholds to the whole batch. To honor **per-stream** thresholds in a shared batch, we pass permissive thresholds (`conf=0.01`, `iou=0.99`) at the model call and re-filter per result using each session's `confidence` and `nms_iou`. The filtering is O(boxes) and negligible compared to forward-pass cost.

### Why max-imgsz per batch

Ultralytics accepts one `imgsz` per call and internally letterboxes all inputs to that size. We use the max `imgsz` across the batch so smaller-imgsz streams aren't downsized below their config. If this turns out to hurt quality or throughput, a future change can bucket streams by imgsz into multiple sub-batches; for v1 a single batch is simpler and correct.

---

## API Surface

All endpoints under `/api/streams`. The existing `/api/stream/*` (singular) is **removed**.

### Collection

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/streams` | — | `{streams: [StreamSummary, ...], capacity: 10}` |
| `POST` | `/api/streams` | `StreamCreate` | `StreamDetail` (201). 409 if at cap; 422 if URL invalid. |
| `GET` | `/api/streams/status` | — | `{streams: [StreamStatus, ...]}` — single-call dashboard refresh. |

### Per-stream

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/api/streams/{id}` | — | `StreamDetail` |
| `PATCH` | `/api/streams/{id}` | `StreamPatch` | `StreamDetail`. URL change triggers a capture restart (see below); param changes are picked up by the next inference cycle without restart. |
| `DELETE` | `/api/streams/{id}` | — | `204`. Stops capture thread, removes session. |
| `GET` | `/api/streams/{id}/status` | — | `StreamStatus` |
| `GET` | `/api/streams/{id}/feed` | — | `multipart/x-mixed-replace` MJPEG. |
| `POST` | `/api/streams/{id}/counting/start` | — | `{counting: true}` |
| `POST` | `/api/streams/{id}/counting/stop` | — | `{counting: false}` |
| `POST` | `/api/streams/{id}/counting/reset` | — | `{counting: <unchanged>, wing_count: 0}` |

### Schemas (pydantic)

```python
class StreamCreate(BaseModel):
    url: str                              # rtsp://...
    name: str | None = None
    confidence: float = 0.25              # ge=0, le=1
    nms_iou: float = 0.45                 # ge=0, le=1
    imgsz: int = 640                      # multiple of 32, ge=32
    roi_position: float = 0.7             # ge=0, le=1
    max_distance: int = 50                # ge=1
    max_disappeared: int = 15             # ge=1

class StreamPatch(BaseModel):
    # All optional; at least one required (model_validator).
    url: str | None = None
    name: str | None = None
    confidence: float | None = None
    nms_iou: float | None = None
    imgsz: int | None = None
    roi_position: float | None = None
    max_distance: int | None = None
    max_disappeared: int | None = None

class StreamStatus(BaseModel):
    id: str
    name: str
    state: Literal["connecting", "connected", "reconnecting", "error", "disconnected"]
    wing_count: int
    counting: bool
    fps: float
    error: str | None

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
```

### Stream IDs

Server-generated `uuid.uuid4().hex[:8]`, returned on POST and used in all per-stream URLs.

### Error responses

- `409 Conflict` — POST when 10 streams already exist (body: `{detail: "Capacity reached (10 streams)"}`).
- `404 Not Found` — `{id}` doesn't exist.
- `422 Unprocessable Entity` — invalid URL, invalid imgsz, out-of-range params, empty PATCH.

### PATCH semantics

- Inference/tracker params (`confidence`, `nms_iou`, `imgsz`, `roi_position`, `max_distance`, `max_disappeared`): assigned to the session under its lock. The inference worker reads them on the next cycle. The `WingCounter`'s `roi_y` is recomputed from the new `roi_position`; `max_disappeared` and `max_distance` are forwarded to the existing `CentroidTracker` instance. Wing count is NOT reset.
- `name`: assigned, no side effects.
- `url`: if the new URL differs from the old, the manager calls `session.restart_capture(new_url)`, which stops the current `CaptureThread`, swaps `session.url`, and starts a new `CaptureThread`. Wing count is preserved (`WingCounter` lives on the session, not the capture thread).

### Global `/api/config` reduction

The existing `/api/config` endpoint is reduced to fields that genuinely remain global:
- `model_path`
- `upload_dir`
- `output_dir`

Per-stream fields (`confidence`, `nms_iou`, `imgsz`, `roi_position`, `max_distance`, `max_disappeared`, `rtsp_url`) move out of global config and live on each `StreamSession`. `docker-compose.yml` env vars for these fields are removed.

---

## Capture Thread Behavior

```python
class CaptureThread:
    BACKOFF_SEQUENCE = (2, 4, 8, 16)   # seconds, then stays at 16

    def _run(self):
        backoff_idx = 0
        while not self._stop.is_set():
            self.session.set_state("connecting")
            cap = cv2.VideoCapture(self.session.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.session.set_state("reconnecting", error=f"Could not open {self.session.url}")
                self._sleep_backoff(backoff_idx)
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
            if not self._stop.is_set():
                self._sleep_backoff(backoff_idx)
                backoff_idx = min(backoff_idx + 1, len(self.BACKOFF_SEQUENCE) - 1)

        self.session.set_state("disconnected")
```

Key behaviors:
- `CAP_PROP_BUFFERSIZE=1` forces OpenCV's FFMPEG backend to keep the buffer minimal, so `cap.read()` returns recent frames even when the inference worker is slower than the camera.
- The backoff sleep uses `self._stop.wait(N)` so deletion mid-backoff returns immediately.
- Successful open resets `backoff_idx` to 0, so a transient blip on a stable camera doesn't escalate.
- The `WingCounter` lives on the session, not the capture thread — reconnects don't reset count. The tracker's `max_disappeared` ages out stale IDs from before the outage.

---

## Inference Worker Loop

```python
class InferenceWorker:
    def __init__(self, model, manager, batch_window_ms: int = 30):
        self.model = model
        self.manager = manager
        self.batch_window_ms = batch_window_ms
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            cycle_start = time.time()

            sessions = self.manager.connected_sessions()
            batch, batch_meta = [], []
            for s in sessions:
                frame = s.take_latest_frame()
                if frame is not None:
                    batch.append(frame)
                    batch_meta.append(s)

            if batch:
                imgsz = max(s.imgsz for s in batch_meta)
                results = self.model(
                    batch, conf=0.01, iou=0.99, imgsz=imgsz, verbose=False
                )
                for session, result in zip(batch_meta, results):
                    det_info = self._extract_dets(result, session.confidence, session.nms_iou)
                    session.process_detections(det_info)

            elapsed_ms = (time.time() - cycle_start) * 1000
            sleep_ms = max(0, self.batch_window_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)
```

`_extract_dets(result, conf_thresh, iou_thresh)`:
- Iterates `result.boxes`, drops boxes with `conf < conf_thresh`.
- Applies per-stream NMS at `iou_thresh` using `torchvision.ops.nms` (or equivalent).
- Returns the same `list[dict]` shape (`x1,y1,x2,y2,conf`) the existing `WingCounter.update()` expects.

`session.process_detections(det_info)`:
1. If `counting`: `objects = counter.update(det_info)`. Else: `objects = counter.tracker.update(...)` (track-only path, same as today).
2. Build the `flash_with_frame` list and call `annotate_detections(...)` from `app/core/annotator.py`.
3. JPEG-encode at quality 80 and store under the session lock.
4. Update sliding-window `fps_display`.

---

## StreamManager

```python
class StreamManager:
    CAPACITY = 10

    def __init__(self, model):
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.RLock()
        self._worker = InferenceWorker(model, self)

    def add(self, create: StreamCreate) -> StreamSession:
        with self._lock:
            if len(self._sessions) >= self.CAPACITY:
                raise CapacityExceeded()
            session = StreamSession.from_create(create)
            self._sessions[session.id] = session
            session.start()                          # starts capture thread
            if len(self._sessions) == 1:
                self._worker.start()
            return session

    def remove(self, stream_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(stream_id, None)
            if session is None:
                raise NotFound()
            session.stop()
            if not self._sessions:
                self._worker.stop()

    def get(self, stream_id: str) -> StreamSession:
        with self._lock:
            session = self._sessions.get(stream_id)
            if session is None:
                raise NotFound()
            return session

    def list(self) -> list[StreamSession]:
        with self._lock:
            return list(self._sessions.values())

    def connected_sessions(self) -> list[StreamSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.state == "connected"]

    def shutdown(self) -> None:
        with self._lock:
            for s in self._sessions.values():
                s.stop()
            self._sessions.clear()
            self._worker.stop()
```

---

## Frontend (`stream.html` + `js/stream.js`)

The page becomes a **stream dashboard**.

**Top of page:** an "Add stream" card with a URL input, name field, and "Add" button. Button is disabled when 10 streams already exist; label changes to "Capacity reached".

**Below:** a responsive CSS grid (`repeat(auto-fill, minmax(360px, 1fr))`) of `StreamCard` elements, one per stream.

**Each card:**
- Header: stream name (editable inline → PATCH on blur), state pill (green/amber/red dot + label), delete button (×) → DELETE confirmation
- MJPEG `<img src="/api/streams/{id}/feed">` (16:9, `object-fit: cover`)
- Stats: wing count (large), FPS (small)
- Controls: Start counting / Stop counting / Reset / Settings (gear → inline panel with sliders for ROI, confidence, nms_iou, imgsz, max_distance, max_disappeared; each input PATCHes `/api/streams/{id}` on change)

**Polling:** single `setInterval(1000ms)` hits `GET /api/streams/status` and fans the response out to every card's state/count/fps. Avoids N parallel polls.

**CSS additions to `style.css`:** `.stream-grid`, `.stream-card`, `.state-pill` (`.state-pill.connected`, `.reconnecting`, `.error`), `.add-stream-card`, settings panel styles.

`index.html` and `video.html` are unchanged.

---

## File Layout

**New files**
- `app/core/stream_session.py` — `StreamSession`
- `app/core/capture_thread.py` — `CaptureThread`
- `app/core/inference_worker.py` — `InferenceWorker`
- `app/core/stream_manager.py` — `StreamManager`, `CapacityExceeded`, `NotFound` exceptions
- `app/routers/streams.py` — collection-based router

**Modified files**
- `app/main.py` — instantiate `StreamManager` in lifespan, store on `app.state`, include new router, drop old `stream` import
- `app/config.py` — drop per-stream fields (`rtsp_url`, `confidence`, `nms_iou`, `imgsz`, `roi_position`, `max_distance`, `max_disappeared`); keep `model_path`, `upload_dir`, `output_dir`
- `app/core/runtime_config.py` — match new `Settings` shape
- `app/routers/config_router.py` — `ConfigResponse`/`ConfigPatch` reduced to the three remaining global fields
- `app/static/stream.html` — replaced with dashboard layout
- `app/static/js/stream.js` — replaced with grid-based logic
- `app/static/css/style.css` — new styles for dashboard
- `docker-compose.yml` — drop the per-stream env vars; keep `MODEL_PATH`

**Deleted files**
- `app/routers/stream.py` (superseded by `streams.py`)

The existing `VideoProcessor` (`app/core/video_processor.py`) remains for the upload-video flow.

---

## Testing Strategy

**Unit tests** (`tests/unit/`)
- `test_stream_session.py`: `take_latest_frame()` returns-and-clears; `push_capture_frame()` overwrites; `process_detections()` updates count only when `counting` is true.
- `test_capture_thread.py`: backoff sequence is `2,4,8,16,16,...`; interruptible sleep wakes on `stop()`; backoff_idx resets after successful open.
- `test_inference_worker.py` (with fake model): batches frames from N sessions, fans results to correct sessions by index, skips cycles with empty batches.
- `test_stream_manager.py`: capacity enforcement (11th add raises), session cleanup on remove, worker start/stop tied to first/last session.

**Integration tests** (`tests/integration/`)
- `test_streams_api.py` (FastAPI `TestClient`):
  - POST 10 streams succeeds, 11th returns 409
  - PATCH with no fields → 422
  - GET unknown id → 404
  - GET /status returns all streams
  - DELETE removes and stops capture thread
  - Counting start/stop/reset toggles state correctly
- `test_end_to_end_file.py`: spin up 3 sessions backed by `cv2.VideoCapture(<short.mp4>)` (same `CaptureThread`, different source), verify each session's `wing_count` reflects its own video, not bleed-through.

**Manual smoke on VPS**
- 10 RTSP streams from a public test feed for 30 minutes
- Verify each card updates independently
- Kill one camera (block ports / unplug); observe `state="reconnecting"`
- Restore camera; observe `state="connected"` and continued counting without count loss

---

## Migration & Rollout

- No data migration required. Existing `app/uploads/` and `app/outputs/` are untouched.
- The old single-stream env vars (`RTSP_URL`, `CONFIDENCE`, `ROI_POSITION`, etc.) are removed from `docker-compose.yml`. Users supply RTSP URLs through the API.
- Container build: no new system dependencies (already has ffmpeg, libgl1, libglib2.0-0).
- The container's single uvicorn worker is retained — the inference worker is a Python thread inside that process and depends on the model being loaded once. Adding uvicorn workers would defeat the shared-model design.

---

## Risks & Open Items

- **GPU memory under batch=10**: a single batched forward pass at imgsz=640 with 10 frames of YOLOv8 is well within a typical VPS GPU's VRAM (~1.5–2 GB per model), but should be confirmed during the VPS smoke test.
- **Heterogeneous imgsz in one batch** may produce suboptimal accuracy for the smaller-imgsz streams. Mitigation: bucket by imgsz into sub-batches if smoke testing reveals issues.
- **Slow streams (~5fps) batched with fast streams (~30fps)** is handled by the single-slot capture buffer: slow streams simply skip cycles. No special logic needed.
