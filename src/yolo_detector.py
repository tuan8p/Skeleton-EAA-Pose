"""YOLO26 detect person: tra ve toi da 2 box conf cao nhat. Fallback None neu loi."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config_manager import ConfigManager

PERSON_CLASS_ID = 0


@dataclass
class PersonBox:
    xyxy: tuple[int, int, int, int]
    confidence: float
    track_id: int | None = None

    def crop(self, frame: np.ndarray, margin: float = 0.1) -> tuple[np.ndarray, tuple[int, int]]:
        """Tra ve (anh_crop, (x0, y0)) voi x0,y0 la offset goc trai-tren tren frame goc."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.xyxy
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x0, y0 = max(0, x1 - mx), max(0, y1 - my)
        return frame[y0:min(h, y2 + my), x0:min(w, x2 + mx)], (x0, y0)


class YOLODetector:
    """Tra ve None o constructor neu khong load duoc model (fallback BlazePose)."""

    def __init__(self, cfg: ConfigManager):
        self.conf_threshold = float(cfg.get("yolo.conf_threshold", 0.5))
        self.iou_threshold = float(cfg.get("yolo.iou_threshold", 0.45))
        self.max_detections = int(cfg.get("yolo.max_detections", 2))
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(cfg.get("yolo.model_name", "yolo26n.pt")))
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray,
               imgsz: int | None = None,
               augment: bool = False) -> list[PersonBox]:
        """Detect person trong frame.

        Args:
            frame_bgr: ảnh BGR numpy array (kích thước gốc).
            imgsz: override inference size (None = để YOLO tự quyết, thường 640).
                   Dùng 1280 khi retry để cải thiện detect người nhỏ.
            augment: bật Test Time Augmentation (flip, scale) khi retry.
        """
        if self._model is None:
            return []
        kwargs: dict = dict(
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            augment=augment,
        )
        if augment:
            kwargs["augment"] = True
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        results = self._model.predict(frame_bgr, **kwargs)
        boxes: list[PersonBox] = []
        if not results or results[0].boxes is None:
            return boxes
        for b in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            boxes.append(PersonBox(xyxy=(x1, y1, x2, y2), confidence=float(b.conf[0])))
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes[: self.max_detections]

    @staticmethod
    def match_by_iou(prev: list[PersonBox], curr: list[PersonBox]) -> list[PersonBox]:
        """Sap xep curr sao cho thu tu person nhat quan voi prev (greedy IoU)."""
        if not prev or not curr:
            return curr
        def iou(a: PersonBox, b: PersonBox) -> float:
            ax1, ay1, ax2, ay2 = a.xyxy
            bx1, by1, bx2, by2 = b.xyxy
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / union if union > 0 else 0.0
        ordered: list[PersonBox] = []
        remaining = list(curr)
        for p in prev:
            if not remaining:
                break
            best = max(remaining, key=lambda c: iou(p, c))
            ordered.append(best)
            remaining.remove(best)
        ordered.extend(remaining)
        return ordered
