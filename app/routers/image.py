import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response

from app.core.runtime_config import runtime_config
from app.core.model_cache import get_model
from app.core.detector import detect_frame
from app.core.annotator import annotate_image_detections

router = APIRouter(prefix="/api/image", tags=["image"])


@router.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return Response(content="Invalid image", status_code=400)

    snap = runtime_config.snapshot()
    model = get_model(snap["model_path"])
    det_info = detect_frame(model, frame, snap["confidence"], snap["nms_iou"], snap["imgsz"])
    annotated, wing_count = annotate_image_detections(frame, det_info)

    _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Wing-Count": str(wing_count),
            "Access-Control-Expose-Headers": "X-Wing-Count",
        },
    )
