"""YOLO person detector: returns up to 2 highest-confidence boxes. Falls back to None on error."""
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
        """Returns (cropped_image, (x0, y0)) where x0,y0 is top-left offset in original frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.xyxy
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x0, y0 = max(0, x1 - mx), max(0, y1 - my)
        return frame[y0:min(h, y2 + my), x0:min(w, x2 + mx)], (x0, y0)


class YOLODetector:
    """Returns None in constructor if model fails to load (BlazePose fallback)."""

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
            frame_bgr: BGR numpy array image (original size).
            imgsz: override inference size (None = YOLO decides, usually 640).
                   Use 1280 on retry to improve small person detection.
            augment: enable Test Time Augmentation (flip, scale) when retrying.
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

    def detect_batch(self, frames_bgr: list[np.ndarray],
                     batch_size: int = 16,
                     imgsz: int | None = None) -> list[list[PersonBox]]:
        """Detect person tren mot batch frames de toi uu VRAM/toc do.

        Args:
            frames_bgr: Danh sach cac anh BGR numpy array.
            batch_size: So luong anh xu ly cung luc (tu config).
            imgsz: Kich thuoc inference.
        """
        if self._model is None or not frames_bgr:
            return [[] for _ in range(len(frames_bgr))]
        
        kwargs: dict = dict(
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            augment=False,
            batch=batch_size,
            stream=True,  # Toi uu bo nho khi tra ve nhieu ket qua
        )
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
            
        # stream=True tra ve generator
        results_gen = self._model.predict(frames_bgr, **kwargs)
        
        batch_boxes: list[list[PersonBox]] = []
        for result in results_gen:
            boxes: list[PersonBox] = []
            if result.boxes is not None:
                for b in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    boxes.append(PersonBox(xyxy=(x1, y1, x2, y2), confidence=float(b.conf[0])))
                boxes.sort(key=lambda b: b.confidence, reverse=True)
                boxes = boxes[: self.max_detections]
            batch_boxes.append(boxes)
            
        return batch_boxes

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
