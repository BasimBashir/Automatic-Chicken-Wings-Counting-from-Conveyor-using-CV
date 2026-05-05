import dataclasses
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.exporter import exporter, ExportState
from app.core.runtime_config import runtime_config
from app.core.model_cache import preload_model

router = APIRouter(prefix="/api/export", tags=["export"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    half: bool = True


class ExportStatusResponse(BaseModel):
    state:        str
    source_model: Optional[str]  = None
    output_path:  Optional[str]  = None
    error:        Optional[str]  = None
    started_at:   Optional[float] = None
    finished_at:  Optional[float] = None
    elapsed_s:    Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/tensorrt",
    response_model=ExportStatusResponse,
    status_code=202,
    summary="Start TensorRT export",
)
def start_tensorrt_export(body: ExportRequest = ExportRequest()) -> dict:
    """Convert the current model to a TensorRT engine in the background.

    - Uses the **model_path** and **imgsz** from the live config.
    - Returns HTTP 202 immediately; poll GET /api/export/tensorrt for status.
    - Returns HTTP 409 if an export is already in progress.
    - When the export completes (state=DONE), a subsequent GET will
      automatically switch **model_path** in the live config to the new engine.

    **Run this on the deployment machine** (inside the GPU container) so the
    engine is compiled for the exact CUDA version that will run inference.
    """
    snap = runtime_config.snapshot()
    started = exporter.start(
        model_path=snap["model_path"],
        imgsz=snap["imgsz"],
        half=body.half,
    )
    if not started:
        raise HTTPException(status_code=409, detail="Export already running — poll GET /api/export/tensorrt")

    return _status_response(exporter.get_status())


@router.get(
    "/tensorrt",
    response_model=ExportStatusResponse,
    summary="Get TensorRT export status",
)
def get_tensorrt_export_status() -> dict:
    """Poll the TensorRT export state.

    When state transitions to **DONE** this endpoint automatically:
    1. Loads the new engine into the model cache.
    2. Updates **model_path** in the live config so all new sessions use it.

    This auto-switch is idempotent — repeated GET calls after DONE are safe.
    """
    status = exporter.get_status()

    if status.state == ExportState.DONE and status.output_path:
        current_path = runtime_config.model_path
        if current_path != status.output_path:
            preload_model(status.output_path)
            runtime_config.update({"model_path": status.output_path})

    return _status_response(status)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_response(status) -> dict:
    d = dataclasses.asdict(status)
    if d["started_at"] and d["finished_at"]:
        d["elapsed_s"] = round(d["finished_at"] - d["started_at"], 1)
    elif d["started_at"]:
        import time
        d["elapsed_s"] = round(time.time() - d["started_at"], 1)
    else:
        d["elapsed_s"] = None
    return d
