"""Compare counts across different tracker configs to localize the issue."""
import sys
import cv2
import time
from collections import Counter

from app.core.detector import load_model, detect_frame
from app.core.counter import WingCounter
from app.core.line_counter import LineCounter
from app.config import settings


def run_one(video_path, label, conf, iou, max_distance, max_disappeared, roi_frac,
            counter_kind="tracker", x_dedup_window=30, dedup_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_y = int(height * roi_frac)
    model = load_model(settings.model_path)
    if counter_kind == "tracker":
        counter = WingCounter(roi_y=roi_y, max_disappeared=max_disappeared,
                              max_distance=max_distance)
    else:
        counter = LineCounter(roi_y=roi_y,
                              x_dedup_window=x_dedup_window,
                              dedup_frames=dedup_frames)

    n = 0
    avg_tracks = 0
    max_tracks = 0
    avg_dets = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        det_info = detect_frame(model, frame, conf=conf, iou=iou)
        objects = counter.update(det_info)
        avg_dets += len(det_info)
        avg_tracks += len(objects)
        max_tracks = max(max_tracks, len(objects))

    cap.release()
    avg_dets /= max(n, 1)
    avg_tracks /= max(n, 1)
    unique_ids = counter.tracker.next_id if hasattr(counter, "tracker") else "-"
    print(f"[{label:38}] count={counter.total_count:4d}  "
          f"avg_dets={avg_dets:.1f}  avg_tracks={avg_tracks:.1f}  "
          f"max_tracks={max_tracks}  unique_ids={unique_ids}  "
          f"runtime={time.time()-t0:.1f}s")
    return counter.total_count


if __name__ == "__main__":
    video = sys.argv[1]
    roi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

    configs = [
        ("tracker max_dis=10",            "tracker", 0.25, 0.45, 40, 10, 30, 8),
        ("tracker max_dis=30",            "tracker", 0.25, 0.45, 40, 30, 30, 8),
        ("tracker max_dis=50",            "tracker", 0.25, 0.45, 40, 50, 30, 8),
        ("tracker max_dis=100",           "tracker", 0.25, 0.45, 40, 100, 30, 8),
        ("trackerless dedup=30/8f",       "line",    0.25, 0.45, 40, 0, 30, 8),
        ("trackerless dedup=20/5f",       "line",    0.25, 0.45, 40, 0, 20, 5),
        ("trackerless dedup=40/15f",      "line",    0.25, 0.45, 40, 0, 40, 15),
        ("trackerless dedup=50/30f",      "line",    0.25, 0.45, 40, 0, 50, 30),
    ]
    for label, kind, c, i, md, mdis, xdw, dfr in configs:
        run_one(video, label, c, i, md, mdis, roi,
                counter_kind=kind, x_dedup_window=xdw, dedup_frames=dfr)
