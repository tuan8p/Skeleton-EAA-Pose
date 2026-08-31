"""Assign consistent IDs to up to 2 persons across frames (PKU 2-person video).

Supports 2 modes (config yolo.tracker):
- "iou":        match by IoU with previous frame (default, no extra library needed)
- "deepocsort": DeepOCSORT (boxmot) + ReID; falls back to IoU if not loadable
"""
from __future__ import annotations

import numpy as np

from .config_manager import ConfigManager
from .yolo_detector import PersonBox, YOLODetector


class BasePersonTracker:
    def reset(self) -> None:
        ...

    def update(self, boxes: list[PersonBox], frame: np.ndarray) -> list[PersonBox]:
        """Return boxes with track_id assigned, sorted into stable person slots."""
        raise NotImplementedError


class IoUTracker(BasePersonTracker):
    def __init__(self):
        self._prev: list[PersonBox] = []

    def reset(self) -> None:
        self._prev = []

    def update(self, boxes: list[PersonBox], frame: np.ndarray) -> list[PersonBox]:
        if not boxes:
            return []
        if not self._prev:
            # First frame: sort left to right
            ordered = sorted(boxes, key=lambda b: b.xyxy[0])
            self._prev = ordered
            return ordered
        ordered = YOLODetector.match_by_iou(self._prev, boxes)
        self._prev = ordered
        return ordered


class DeepOCSORTTracker(BasePersonTracker):
    """DeepOCSORT qua boxmot (>=19: dung create_tracker). Slot person = thu tu
    track_id xuat hien dau tien trong chunk."""

    def __init__(self, cfg: ConfigManager):
        from pathlib import Path
        from boxmot.trackers.registry import create_tracker
        reid = Path(str(cfg.get("yolo.reid_weights", "osnet_x0_25_msmt17.pt")))
        self._tracker = create_tracker(
            "deepocsort",
            reid_weights=reid,
            device="cpu",
            half=False,
            tracker_kwargs={"det_thresh": float(cfg.get("yolo.conf_threshold", 0.2))},
        )
        self._slot_of_track: dict[int, int] = {}

    def reset(self) -> None:
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
        self._slot_of_track = {}

    def update(self, boxes: list[PersonBox], frame: np.ndarray) -> list[PersonBox]:
        if not boxes:
            return []
        dets = np.array([[*b.xyxy, b.confidence, 0] for b in boxes], dtype=np.float32)
        tracks = self._tracker.update(dets, frame)
        if tracks is None or len(tracks) == 0:
            return []
        out: list[PersonBox] = []
        for trk in np.atleast_2d(tracks):
            x1, y1, x2, y2, track_id, conf = (float(v) for v in trk[:6])
            tid = int(track_id)
            if tid not in self._slot_of_track:
                if len(self._slot_of_track) >= 2:
                    continue
                self._slot_of_track[tid] = len(self._slot_of_track)
            out.append(PersonBox(xyxy=(int(x1), int(y1), int(x2), int(y2)),
                                 confidence=conf, track_id=tid))
        out.sort(key=lambda b: self._slot_of_track[b.track_id])
        return out[:2]


def get_person_tracker(cfg: ConfigManager) -> BasePersonTracker:
    name = str(cfg.get("yolo.tracker", "iou")).lower()
    if name == "deepocsort":
        try:
            return DeepOCSORTTracker(cfg)
        except Exception:
            pass  # fallback IoU khi boxmot/weights khong co
    return IoUTracker()
