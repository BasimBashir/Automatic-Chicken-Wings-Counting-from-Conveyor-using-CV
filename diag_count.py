"""Diagnostic: run the full detection/tracking/counting pipeline on a video
and report what the counter is doing each frame so we can see where counts
are missed, duplicated, or inflated."""
import sys
import cv2
from collections import Counter

from app.core.detector import load_model, detect_frame
from app.core.counter import WingCounter
from app.core.annotator import annotate_detections
from app.config import settings


def run(video_path: str, roi_frac: float = 0.7, out_path: str = "diag_out.mp4",
        conf: float = 0.25, iou: float = 0.45,
        max_distance: int = 40, max_disappeared: int = 50,
        log_every: int = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_y = int(height * roi_frac)
    print(f"video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, roi_y={roi_y}")

    model = load_model(settings.model_path)
    counter = WingCounter(roi_y=roi_y, max_disappeared=max_disappeared,
                          max_distance=max_distance)
    counter.is_counting = True  # always count

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    prev_track_ids = set()
    prev_count = 0
    crossings = []
    track_lifespans = Counter()  # id -> num frames seen
    track_first_seen = {}
    track_last_bbox = {}
    detections_per_frame = []
    tracks_per_frame = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        det_info = detect_frame(model, frame, conf=conf, iou=iou)
        detections_per_frame.append(len(det_info))

        objects = counter.update(det_info)
        tracks_per_frame.append(len(objects))

        cur_track_ids = set(objects.keys())
        new_tracks = cur_track_ids - prev_track_ids
        lost_tracks = prev_track_ids - cur_track_ids

        for t in cur_track_ids:
            track_lifespans[t] += 1
            track_first_seen.setdefault(t, frame_num)
            if t in counter.tracker.bboxes:
                track_last_bbox[t] = counter.tracker.bboxes[t]

        if counter.total_count != prev_count:
            new_count = counter.total_count - prev_count
            crossing_ids = list(counter.counted_ids)[-new_count:]
            crossings.append((frame_num, new_count, crossing_ids))
            print(f"  f{frame_num}: +{new_count} count "
                  f"(total={counter.total_count}, crossing ids={crossing_ids})")
            prev_count = counter.total_count

        if frame_num % log_every == 0 or new_tracks or lost_tracks:
            extras = []
            if new_tracks:
                extras.append(f"new={sorted(new_tracks)}")
            if lost_tracks:
                extras.append(f"lost={sorted(lost_tracks)}")
            if extras:
                print(f"f{frame_num:4d} dets={len(det_info)} tracks={len(objects)} "
                      + " ".join(extras))

        flash_with_frame = [(fx, fy, frame_num - i)
                            for i, (fx, fy) in enumerate(
                                reversed(counter.flash_events[-12:]))]
        annotated = annotate_detections(
            frame=frame, detections=det_info, objects=objects,
            counted_ids=counter.counted_ids, trails=counter.trails,
            flash_events=flash_with_frame, roi_y=counter.roi_y,
            frame_num=frame_num, total_count=counter.total_count,
            total_frames=total_frames, is_stream=False, fps_display=fps,
        )
        writer.write(annotated)

        prev_track_ids = cur_track_ids

    cap.release()
    writer.release()

    print()
    print("=" * 60)
    print(f"FINAL COUNT: {counter.total_count}")
    print(f"frames processed: {frame_num}")
    print(f"avg detections/frame: {sum(detections_per_frame)/max(frame_num,1):.2f}")
    print(f"max detections in one frame: {max(detections_per_frame) if detections_per_frame else 0}")
    print(f"avg active tracks/frame: {sum(tracks_per_frame)/max(frame_num,1):.2f}")
    print(f"max active tracks: {max(tracks_per_frame) if tracks_per_frame else 0}")
    print(f"unique track IDs ever created: {len(track_lifespans)}")
    print(f"counted (crossed line): {len(counter.counted_ids)}")

    short_tracks = [t for t, n in track_lifespans.items() if n <= 3]
    print(f"tracks lasting <=3 frames: {len(short_tracks)} (potential ID swaps/flicker)")

    print("\ntracks that crossed the line:")
    for tid in sorted(counter.counted_ids):
        first = track_first_seen.get(tid)
        life = track_lifespans.get(tid)
        last_bb = track_last_bbox.get(tid)
        print(f"  id={tid}  first_seen=f{first}  lifespan={life} frames  last_bbox={last_bb}")

    print("\ntracks that did NOT cross (top 20 by lifespan):")
    uncounted = [(tid, n) for tid, n in track_lifespans.items()
                 if tid not in counter.counted_ids]
    uncounted.sort(key=lambda x: -x[1])
    for tid, life in uncounted[:20]:
        first = track_first_seen.get(tid)
        last_bb = track_last_bbox.get(tid)
        print(f"  id={tid}  first_seen=f{first}  lifespan={life}  last_bbox={last_bb}")

    print(f"\nannotated output: {out_path}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video:
        print("usage: python diag_count.py <video_path> [roi_frac]")
        sys.exit(1)
    roi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    run(video, roi_frac=roi)
