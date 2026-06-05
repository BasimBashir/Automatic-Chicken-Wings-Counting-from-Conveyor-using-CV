import cv2
import numpy as np

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


def annotate_detections(frame, detections, objects, counted_ids, trails,
                        flash_events, roi_y, frame_num, total_count,
                        total_frames, is_stream, fps_display):
    """Full-frame annotation: bboxes, trails, ROI line, flashes."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for obj_id, trail_pts in trails.items():
        if obj_id in counted_ids:
            draw_trail(annotated, trail_pts, COLORS["counted"])
        else:
            draw_trail(annotated, trail_pts, COLORS["trail"])

    for info in detections:
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        is_counted = False
        for obj_id, (ox, oy) in objects.items():
            if abs(ox - cx) < 5 and abs(oy - cy) < 5:
                is_counted = obj_id in counted_ids
                break
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], is_counted, info["conf"])

    if roi_y is not None:
        draw_roi_line(annotated, roi_y, width, frame_num)

    active_flashes = []
    for (fx, fy, f_start) in flash_events:
        age = frame_num - f_start
        if age < 12:
            intensity = 1.0 - (age / 12.0)
            draw_crossing_flash(annotated, fx, fy, intensity)
            active_flashes.append((fx, fy, f_start))
    flash_events.clear()
    flash_events.extend(active_flashes)

    for obj_id, (cx, cy) in objects.items():
        color = COLORS["counted"] if obj_id in counted_ids else COLORS["uncounted"]
        cv2.circle(annotated, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)

    if roi_y is not None:
        cv2.putText(annotated, "COUNTING LINE", (width - 200, roi_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["roi_line"], 2, cv2.LINE_AA)

    return annotated


def annotate_image_detections(frame, det_info):
    """Annotate a single image with detection boxes and count overlay."""
    annotated = frame.copy()
    wing_count = len(det_info)

    for info in det_info:
        draw_bbox(annotated, info["x1"], info["y1"],
                  info["x2"], info["y2"], counted=False, conf=info["conf"])
        cx = (info["x1"] + info["x2"]) // 2
        cy = (info["y1"] + info["y2"]) // 2
        cv2.circle(annotated, (cx, cy), 4, COLORS["accent"], -1, cv2.LINE_AA)

    draw_rounded_rect(annotated, (8, 8), (280, 55), COLORS["panel_bg"], radius=8, alpha=0.85)
    cv2.putText(annotated, f"{wing_count}", (18, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, COLORS["counted"], 3, cv2.LINE_AA)
    tw = cv2.getTextSize(f"{wing_count}", cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0][0]
    cv2.putText(annotated, "wings detected", (18 + tw + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA)

    return annotated, wing_count
