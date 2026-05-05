import os
import uuid
import shutil
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model
from app.core.video_processor import VideoProcessor

router = APIRouter(prefix="/api/video", tags=["video"])

_sessions: dict[str, VideoProcessor] = {}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    snap = runtime_config.snapshot()

    session_id = str(uuid.uuid4())[:8]
    filepath = os.path.join(snap["upload_dir"], f"{session_id}_{file.filename}")

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    height = _get_video_height(filepath)
    roi_y = int(height * snap["roi_position"])
    raw_output = os.path.join(snap["output_dir"], f"{session_id}_raw.mp4")

    model = get_model(snap["model_path"])
    processor = VideoProcessor(
        source=filepath,
        model=model,
        roi_y=roi_y,
        confidence=snap["confidence"],
        nms_iou=snap["nms_iou"],
        imgsz=snap["imgsz"],
        max_disappeared=snap["max_disappeared"],
        max_distance=snap["max_distance"],
        save_raw_path=raw_output,
        is_stream=False,
    )
    _sessions[session_id] = processor
    return {"session_id": session_id, "filename": file.filename}


@router.post("/{session_id}/start")
def start_video(session_id: str):
    _get_session(session_id).start()
    return {"status": "playing"}


@router.post("/{session_id}/stop")
def stop_video(session_id: str):
    _get_session(session_id).stop()
    return {"status": "stopped"}


@router.post("/{session_id}/counting/start")
def start_counting(session_id: str):
    _get_session(session_id).start_counting()
    return {"status": "counting"}


@router.post("/{session_id}/counting/stop")
def stop_counting(session_id: str):
    _get_session(session_id).stop_counting()
    return {"status": "not_counting"}


@router.get("/{session_id}/feed")
def video_feed(session_id: str):
    proc = _get_session(session_id)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{session_id}/status")
def video_status(session_id: str):
    return _get_session(session_id).get_status()


@router.get("/{session_id}/download")
def download_video(session_id: str):
    proc = _get_session(session_id)
    raw_path = proc.save_raw_path
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=404, detail="Output not ready")

    snap = runtime_config.snapshot()
    output_path = os.path.join(snap["output_dir"], f"{session_id}_output.mp4")
    if not os.path.exists(output_path):
        VideoProcessor.reencode_h264(raw_path, output_path)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"wing_count_{session_id}.mp4",
    )


def _get_session(session_id: str) -> VideoProcessor:
    proc = _sessions.get(session_id)
    if proc is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return proc


def _get_video_height(filepath: str) -> int:
    cap = cv2.VideoCapture(filepath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height if height > 0 else 384


def _mjpeg_generator(proc: VideoProcessor):
    import time
    while proc.is_playing or not proc.is_complete:
        frame_bytes = proc.latest_frame
        if frame_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
        time.sleep(0.03)
