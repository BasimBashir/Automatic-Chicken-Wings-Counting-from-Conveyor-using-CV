import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/stream", tags=["stream"])

_processor: VideoProcessor | None = None


class StreamStart(BaseModel):
    url: str | None = None


@router.post("/start")
def start_stream(body: StreamStart = StreamStart()):
    global _processor

    snap = runtime_config.snapshot()
    url = body.url or snap["rtsp_url"]
    if not url:
        raise HTTPException(status_code=400, detail="No RTSP URL provided")

    if _processor and _processor.is_playing:
        _processor.stop()

    model = get_model(snap["model_path"])
    roi_y = int(384 * snap["roi_position"])

    _processor = VideoProcessor(
        source=url,
        model=model,
        roi_y=roi_y,
        confidence=snap["confidence"],
        nms_iou=snap["nms_iou"],
        imgsz=snap["imgsz"],
        max_disappeared=snap["max_disappeared"],
        max_distance=snap["max_distance"],
        is_stream=True,
    )
    _processor.start()
    return {"status": "connected", "url": url}


@router.post("/stop")
def stop_stream():
    global _processor
    if _processor:
        _processor.stop()
    return {"status": "disconnected"}


@router.post("/counting/start")
def start_counting():
    if not _processor or not _processor.is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    _processor.start_counting()
    return {"status": "counting"}


@router.post("/counting/stop")
def stop_counting():
    if not _processor or not _processor.is_playing:
        raise HTTPException(status_code=400, detail="Stream not active")
    _processor.stop_counting()
    return {"status": "not_counting"}


@router.get("/feed")
def stream_feed():
    if not _processor:
        raise HTTPException(status_code=400, detail="Stream not started")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
def stream_status():
    if not _processor:
        return {"is_connected": False, "is_counting": False, "wing_count": 0, "fps": 0}
    status = _processor.get_status()
    return {
        "is_connected": status["is_playing"],
        "is_counting":  status["is_counting"],
        "wing_count":   status["wing_count"],
        "fps":          status["fps"],
        "error":        status["error"],
    }


def _mjpeg_generator():
    while _processor and _processor.is_playing:
        frame_bytes = _processor.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
