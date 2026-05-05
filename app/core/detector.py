import platform
import pathlib
import warnings
import numpy as np
from ultralytics import YOLO

from app.core.model_cache import get_model, preload_model  # noqa: F401 — re-exported for back-compat

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath


def load_model(model_path: str) -> YOLO:
    """Backward-compatible shim — delegates to the shared model cache."""
    return get_model(model_path)


def detect_frame(model: YOLO, frame: np.ndarray, conf: float = 0.25,
                 iou: float = 0.45, imgsz: int = 640) -> list[dict]:
    results = model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    det_info = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_info.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(box.conf[0]),
        })
    return det_info
