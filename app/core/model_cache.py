import threading
from ultralytics import YOLO

_cache: dict[str, YOLO] = {}
_lock = threading.Lock()


def get_model(path: str) -> YOLO:
    """Return cached YOLO model for *path*, loading it on first call.

    Uses double-checked locking: cache hits never acquire the lock, so
    ongoing inference is not blocked while a new model is being loaded.
    """
    if path in _cache:
        return _cache[path]
    with _lock:
        if path not in _cache:
            _cache[path] = YOLO(path)
        return _cache[path]


def preload_model(path: str) -> None:
    """Eagerly load *path* into the cache (used at startup and on model switch)."""
    get_model(path)
