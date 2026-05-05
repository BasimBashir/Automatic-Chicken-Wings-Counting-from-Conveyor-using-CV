from collections import deque
from app.core.tracker import CentroidTracker


def _det_to_tuple(d):
    cx = (d["x1"] + d["x2"]) // 2
    cy = (d["y1"] + d["y2"]) // 2
    return (cx, cy, d["x1"], d["y1"], d["x2"], d["y2"])


class WingCounter:
    """Wraps CentroidTracker with centroid-based ROI line crossing.

    Each tracked bbox has its center point monitored every frame.
    When the center crosses the ROI line top→bottom, count += 1.
    Stacked wings that are tracked as separate IDs each fire their own
    crossing event, so a pile of 5 overlapping bboxes = +5 count.
    """

    def __init__(self, roi_y: int, max_disappeared: int = 15,
                 max_distance: int = 50, trail_length: int = 18):
        self.roi_y = roi_y
        self.tracker = CentroidTracker(max_disappeared=max_disappeared,
                                       max_distance=max_distance)
        self.trail_length = trail_length
        self.total_count = 0
        self.counted_ids = set()
        self.last_cy = {}   # obj_id -> previous centroid y
        self.trails = {}
        self.flash_events = []

    def update(self, det_info: list[dict]) -> dict:
        """Process detections for one frame. Returns current tracked objects."""
        detections = [_det_to_tuple(d) for d in det_info]

        objects = self.tracker.update(detections)
        active_ids = set(objects.keys())

        for obj_id, (cx, cy) in objects.items():
            if obj_id not in self.trails:
                self.trails[obj_id] = deque(maxlen=self.trail_length)
            self.trails[obj_id].append((int(cx), int(cy)))

        for old_id in list(self.trails.keys()):
            if old_id not in active_ids:
                del self.trails[old_id]

        for obj_id, (cx, cy) in objects.items():
            if obj_id in self.counted_ids:
                continue
            prev_cy = self.last_cy.get(obj_id)
            self.last_cy[obj_id] = cy

            if prev_cy is None:
                # New track: if it first appears at or below the ROI it has
                # already crossed the line (wing entered frame mid-lane or
                # was detected late). Count it immediately.
                if cy >= self.roi_y:
                    self.total_count += 1
                    self.counted_ids.add(obj_id)
                    self.flash_events.append((int(cx), int(cy)))
            elif prev_cy < self.roi_y <= cy:
                # Centroid crossed the ROI line top→bottom this frame.
                self.total_count += 1
                self.counted_ids.add(obj_id)
                self.flash_events.append((int(cx), int(cy)))

        for old_id in list(self.last_cy.keys()):
            if old_id not in active_ids:
                del self.last_cy[old_id]

        # Retire counted tracks immediately on first disappearance so their ID
        # is not reused for the next wing arriving in the same area.
        for cid in list(self.counted_ids):
            if cid in self.tracker.disappeared and self.tracker.disappeared[cid] > 0:
                self.tracker._deregister(cid)

        return objects

    def reset(self):
        self.tracker.reset()
        self.total_count = 0
        self.counted_ids = set()
        self.last_cy = {}
        self.trails = {}
        self.flash_events = []
