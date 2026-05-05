"""Trackerless line counter for one-way conveyors.

Each frame: any detection whose bbox straddles the ROI line is a candidate.
A candidate counts iff no detection was already counted near the same x
within the last `dedup_frames` frames. No IDs, no Hungarian assignment.

Suited to: one-way conveyor (top->bottom), ROI near the exit, wings can be
back-to-back as long as they're not in the exact same lane simultaneously.
"""
from collections import deque


class LineCounter:
    def __init__(self, roi_y: int, x_dedup_window: int = 30,
                 dedup_frames: int = 8, trail_length: int = 18):
        self.roi_y = roi_y
        self.x_dedup_window = x_dedup_window
        self.dedup_frames = dedup_frames
        self.trail_length = trail_length
        self.total_count = 0
        self.frame_num = 0
        # recent (frame_num, x_center) of counted crossings
        self.recent = deque()
        # purely cosmetic, for the existing annotator
        self.counted_ids = set()
        self.trails = {}
        self.flash_events = []

    def update(self, det_info: list[dict]) -> dict:
        self.frame_num += 1

        # Drop expired dedup entries
        cutoff = self.frame_num - self.dedup_frames
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()

        objects = {}  # synthetic id -> (cx, cy) for annotator compatibility
        for i, d in enumerate(det_info):
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2
            objects[self.frame_num * 10000 + i] = (cx, cy)

            crosses_line = d["y1"] <= self.roi_y <= d["y2"]
            if not crosses_line:
                continue

            recently_counted = any(
                abs(rx - cx) < self.x_dedup_window for _, rx in self.recent
            )
            if recently_counted:
                continue

            self.total_count += 1
            self.recent.append((self.frame_num, cx))
            self.flash_events.append((cx, cy))

        return objects

    def reset(self):
        self.total_count = 0
        self.frame_num = 0
        self.recent.clear()
        self.counted_ids.clear()
        self.trails.clear()
        self.flash_events.clear()
