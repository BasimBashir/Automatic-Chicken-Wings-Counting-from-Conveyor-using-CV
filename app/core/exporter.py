import dataclasses
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional


class ExportState(str, Enum):
    IDLE    = "IDLE"
    RUNNING = "RUNNING"
    DONE    = "DONE"
    FAILED  = "FAILED"


@dataclasses.dataclass
class ExportStatus:
    state:        ExportState      = ExportState.IDLE
    source_model: Optional[str]    = None
    output_path:  Optional[str]    = None
    error:        Optional[str]    = None
    started_at:   Optional[float]  = None
    finished_at:  Optional[float]  = None


class TensorRTExporter:
    """Background TensorRT export with a strict IDLE→RUNNING→DONE/FAILED state machine.

    The _lock is held only for status reads/writes — never during the export
    itself (which can take several minutes), so status polls are never blocked.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = ExportStatus()

    def get_status(self) -> ExportStatus:
        with self._lock:
            return dataclasses.replace(self._status)

    def start(self, model_path: str, imgsz: int, half: bool = True) -> bool:
        """Begin export in a daemon thread. Returns False if already RUNNING."""
        with self._lock:
            if self._status.state == ExportState.RUNNING:
                return False
            self._status = ExportStatus(
                state=ExportState.RUNNING,
                source_model=model_path,
                started_at=time.time(),
            )
        thread = threading.Thread(
            target=self._run,
            args=(model_path, imgsz, half),
            daemon=True,
        )
        thread.start()
        return True

    def _run(self, model_path: str, imgsz: int, half: bool) -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            result = model.export(format="engine", imgsz=imgsz, half=half)
            output = str(result) if result else str(Path(model_path).with_suffix(".engine"))
            with self._lock:
                self._status = ExportStatus(
                    state=ExportState.DONE,
                    source_model=model_path,
                    output_path=output,
                    started_at=self._status.started_at,
                    finished_at=time.time(),
                )
        except Exception as exc:
            with self._lock:
                self._status = ExportStatus(
                    state=ExportState.FAILED,
                    source_model=model_path,
                    error=str(exc),
                    started_at=self._status.started_at,
                    finished_at=time.time(),
                )


exporter = TensorRTExporter()
