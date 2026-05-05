import platform
import cv2
import numpy as np
import pathlib
import argparse
import warnings
import time
from collections import OrderedDict, deque
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

MODEL_PATH = r"runs\detect\runs\chicken_wings\weights\best.pt"

# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    "panel_bg":     (20, 20, 20),
    "panel_border": (60, 60, 60),
    "accent":       (0, 200, 255),
    "counted":      (0, 230, 118),
    "uncounted":    (255, 180, 0),
    "roi_line":     (80, 80, 255),
    "roi_glow":     (60, 60, 200),
    "flash":        (0, 255, 255),
    "bbox":         (255, 170, 50),
    "bbox_counted": (0, 200, 100),
    "trail":        (200, 120, 0),
    "white":        (255, 255, 255),
    "dim":          (160, 160, 160),
    "very_dim":     (100, 100, 100),
}


class CentroidTracker:
    """Centroid-based object tracker for counting objects crossing a line."""

    def __init__(self, max_disappeared=50, max_distance=40):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))
        det_centroids = np.array(detections)

        diff = obj_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = tuple(det_centroids[col])
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(obj_centroids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

        for col in range(len(det_centroids)):
            if col not in used_cols:
                self._register(tuple(det_centroids[col]))

        return self.objects

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1


# ── Drawing Helpers ────────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.85):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_trail(img, points, base_color, max_length=20):
    pts = list(points)
    n = len(pts)
    if n < 2:
        return
    overlay = img.copy()
    for i in range(1, n):
        t = i / n
        r = int(base_color[0] * (0.3 + 0.7 * t))
        g = int(base_color[1] * (0.3 + 0.7 * t))
        b = int(base_color[2] * (0.3 + 0.7 * t))
        color = (r, g, b)
        cv2.line(overlay, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        if i == n - 1:
            cv2.circle(overlay, pts[i], 2, base_color, -1, cv2.LINE_AA)
        elif i % 3 == 0:
            cv2.circle(overlay, pts[i], 1, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)


def draw_roi_line(img, roi_y, width, frame_num):
    cv2.line(img, (0, roi_y), (width, roi_y), COLORS["roi_glow"], 6, cv2.LINE_AA)
    dash_len = 20
    gap_len = 12
    offset = (frame_num * 2) % (dash_len + gap_len)
    x = -offset
    while x < width:
        x1 = max(0, x)
        x2 = min(width, x + dash_len)
        if x2 > x1:
            cv2.line(img, (x1, roi_y), (x2, roi_y), COLORS["roi_line"], 2, cv2.LINE_AA)
        x += dash_len + gap_len
    arrow_spacing = 120
    for ax in range(arrow_spacing // 2, width, arrow_spacing):
        cv2.arrowedLine(
            img, (ax, roi_y - 10), (ax, roi_y + 10),
            COLORS["roi_line"], 2, cv2.LINE_AA, tipLength=0.5
        )


def draw_crossing_flash(img, cx, cy, intensity):
    overlay = img.copy()
    ring_radius = int(8 + 20 * (1.0 - intensity))
    ring_thickness = max(1, int(2 * intensity))
    alpha = intensity * 0.6
    cv2.circle(overlay, (cx, cy), ring_radius, COLORS["counted"], ring_thickness, cv2.LINE_AA)
    if intensity > 0.5:
        inner_r = int(4 * intensity)
        cv2.circle(overlay, (cx, cy), inner_r, COLORS["flash"], -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_dashboard(img, total_count, in_frame, total_tracked, frame_num,
                   total_frames, is_stream, fps_display, width):
    panel_w = 300
    panel_h = 130
    margin = 8
    draw_rounded_rect(
        img, (margin, margin), (margin + panel_w, margin + panel_h),
        COLORS["panel_bg"], radius=10, alpha=0.88
    )
    cv2.line(img, (margin + 10, margin + 2), (margin + panel_w - 10, margin + 2),
             COLORS["accent"], 2, cv2.LINE_AA)
    x0 = margin + 14
    y0 = margin + 30
    count_text = f"{total_count}"
    cv2.putText(img, count_text, (x0, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLORS["counted"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0][0]
    cv2.putText(img, "wings counted", (x0 + tw + 8, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + 18), (x0 + panel_w - 30, y0 + 18),
             COLORS["panel_border"], 1, cv2.LINE_AA)
    y1 = y0 + 42
    cv2.putText(img, f"In Frame: {in_frame}", (x0, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["white"], 1, cv2.LINE_AA)
    cv2.putText(img, f"Tracked: {total_tracked}", (x0 + 140, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["accent"], 1, cv2.LINE_AA)
    y2 = y1 + 24
    if is_stream:
        frame_text = f"Frame: {frame_num}"
    else:
        pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
        frame_text = f"Frame: {frame_num}/{total_frames} ({pct:.0f}%)"
    cv2.putText(img, frame_text, (x0, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["very_dim"], 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps_display:.0f}", (x0 + 200, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["very_dim"], 1, cv2.LINE_AA)
    if not is_stream and total_frames > 0:
        y3 = y2 + 18
        bar_x1 = x0
        bar_x2 = x0 + panel_w - 30
        bar_w = bar_x2 - bar_x1
        progress = frame_num / total_frames
        cv2.rectangle(img, (bar_x1, y3), (bar_x2, y3 + 4), COLORS["panel_border"], -1)
        cv2.rectangle(img, (bar_x1, y3), (bar_x1 + int(bar_w * progress), y3 + 4),
                      COLORS["accent"], -1)


def draw_bbox(img, x1, y1, x2, y2, counted, conf):
    color = COLORS["bbox_counted"] if counted else COLORS["bbox"]
    corner_len = 8
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)
    label = f"{conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)


# ── Core Functions ─────────────────────────────────────────────────────────────

def load_model(model_path=MODEL_PATH):
    """Load the YOLOv8 custom model."""
    model = YOLO(model_path)
    return model


def detect_and_annotate_image(model, image_path, conf_threshold=0.25, save_path=None):
    """Detect chicken wings in a single image, annotate it, and return the count."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None, 0

    results = model(image, conf=conf_threshold, verbose=False)
    result = results[0]
    boxes = result.boxes
    wing_count = len(boxes)

    annotated = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        draw_bbox(annotated, x1, y1, x2, y2, counted=False, conf=conf)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, COLORS["accent"], -1, cv2.LINE_AA)

    draw_rounded_rect(annotated, (8, 8), (280, 55), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(
        annotated, f"{wing_count}", (18, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["counted"], 3, cv2.LINE_AA
    )
    tw = cv2.getTextSize(f"{wing_count}", cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(
        annotated, "wings detected", (18 + tw + 8, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA
    )

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Annotated image saved to '{save_path}'")

    print(f"Detected {wing_count} chicken wing(s) in '{image_path}'")
    return annotated, wing_count


def detect_and_annotate_video(
    model, video_path, conf_threshold=0.25, save_path=None,
    roi_position=0.7, max_disappeared=50, max_distance=40
):
    """Detect and count chicken wings crossing a ROI line in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_stream = total_frames <= 0

    roi_y = int(height * roi_position)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    tracker = CentroidTracker(
        max_disappeared=max_disappeared, max_distance=max_distance
    )
    total_count = 0
    counted_ids = set()
    prev_positions = {}
    trails = {}
    flash_events = []
    frame_num = 0
    trail_length = 18

    fps_timer = time.time()
    fps_display = 0.0
    fps_frame_count = 0

    if is_stream:
        print(f"Processing live stream: {video_path}")
    else:
        print(f"Processing video: {video_path} ({total_frames} frames at {fps:.1f} FPS)")
    print(f"ROI counting line at y={roi_y} ({roi_position*100:.0f}% from top)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_stream:
                print("Stream ended or connection lost.")
            break

        frame_num += 1
        fps_frame_count += 1

        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = time.time()

        results = model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        boxes = result.boxes

        centroids = []
        det_info = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroids.append((cx, cy))
            det_info.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

        objects = tracker.update(centroids)
        active_ids = set(objects.keys())

        for obj_id, (cx, cy) in objects.items():
            if obj_id not in trails:
                trails[obj_id] = deque(maxlen=trail_length)
            trails[obj_id].append((int(cx), int(cy)))

        for old_id in list(trails.keys()):
            if old_id not in active_ids:
                del trails[old_id]

        for obj_id, (cx, cy) in objects.items():
            if obj_id in counted_ids:
                continue
            prev_y = prev_positions.get(obj_id)
            if prev_y is not None:
                if (prev_y >= roi_y > cy) or (prev_y <= roi_y < cy):
                    total_count += 1
                    counted_ids.add(obj_id)
                    flash_events.append((int(cx), int(cy), frame_num))
            prev_positions[obj_id] = cy

        for old_id in list(prev_positions.keys()):
            if old_id not in active_ids:
                del prev_positions[old_id]

        # ── Render ─────────────────────────────────────────────────────────
        annotated = frame.copy()

        for obj_id, trail_pts in trails.items():
            if obj_id in counted_ids:
                draw_trail(annotated, trail_pts, COLORS["counted"], trail_length)
            else:
                draw_trail(annotated, trail_pts, COLORS["trail"], trail_length)

        for info in det_info:
            cx = (info["x1"] + info["x2"]) // 2
            cy = (info["y1"] + info["y2"]) // 2
            is_counted = False
            for obj_id, (ox, oy) in objects.items():
                if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                    is_counted = obj_id in counted_ids
                    break
            draw_bbox(annotated, info["x1"], info["y1"],
                      info["x2"], info["y2"], is_counted, info["conf"])

        draw_roi_line(annotated, roi_y, width, frame_num)

        active_flashes = []
        for (fx, fy, f_start) in flash_events:
            age = frame_num - f_start
            if age < 12:
                intensity = 1.0 - (age / 12.0)
                draw_crossing_flash(annotated, fx, fy, intensity)
                active_flashes.append((fx, fy, f_start))
        flash_events = active_flashes

        for obj_id, (cx, cy) in objects.items():
            if obj_id in counted_ids:
                color = COLORS["counted"]
            else:
                color = COLORS["uncounted"]
            cv2.circle(annotated, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)

        draw_dashboard(
            annotated, total_count, len(objects), tracker.next_id,
            frame_num, total_frames, is_stream, fps_display, width
        )

        if writer:
            writer.write(annotated)

        cv2.imshow("Chicken Wing Counter", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Playback stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Annotated video saved to '{save_path}'")
    cv2.destroyAllWindows()

    print(f"Total chicken wings crossed ROI line: {total_count}")
    return total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chicken Wing Detection and Counting")
    parser.add_argument("input", help="Path to an image, video file, or RTSP stream URL")
    parser.add_argument("--save", default=None, help="Path to save the annotated output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to YOLOv8 model weights")
    parser.add_argument("--roi", type=float, default=0.7,
                        help="ROI line position as fraction of frame height (default: 0.7)")
    parser.add_argument("--max-distance", type=int, default=40,
                        help="Max pixel distance for matching across frames (default: 40)")
    parser.add_argument("--max-disappeared", type=int, default=50,
                        help="Frames before a lost track is dropped (default: 50)")
    args = parser.parse_args()

    model = load_model(args.model)

    is_stream = args.input.startswith("rtsp://") or args.input.startswith("http")

    if is_stream:
        detect_and_annotate_video(
            model, args.input, conf_threshold=args.conf, save_path=args.save,
            roi_position=args.roi, max_distance=args.max_distance,
            max_disappeared=args.max_disappeared,
        )
    else:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        ext = pathlib.Path(args.input).suffix.lower()

        if ext in image_exts:
            annotated, count = detect_and_annotate_image(
                model, args.input, conf_threshold=args.conf, save_path=args.save
            )
            if annotated is not None:
                cv2.imshow("Chicken Wing Detection - Image", annotated)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        elif ext in video_exts:
            detect_and_annotate_video(
                model, args.input, conf_threshold=args.conf, save_path=args.save,
                roi_position=args.roi, max_distance=args.max_distance,
                max_disappeared=args.max_disappeared,
            )

        else:
            print(f"Error: Unsupported file extension '{ext}'")
            print(f"Supported images: {image_exts}")
            print(f"Supported videos: {video_exts}")
