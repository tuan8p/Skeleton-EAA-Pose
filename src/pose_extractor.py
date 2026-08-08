"""Quan ly MediaPipe BlazePose (Tasks API - PoseLandmarker).

mediapipe >= 0.10.30 da bo legacy solutions API (mp.solutions.pose), bat buoc
dung PoseLandmarker voi model .task. Model duoc tro boi mediapipe.model_path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_manager import ConfigManager


@dataclass
class PoseResult:
    world_landmarks: np.ndarray   # (33, 3) met, goc tai mid-hip
    image_landmarks: np.ndarray   # (33, 3) normalized x,y + z tuong doi
    visibility: np.ndarray        # (33,)


class PoseExtractor:
    def __init__(self, cfg: ConfigManager):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (PoseLandmarker,
                                                   PoseLandmarkerOptions,
                                                   RunningMode)
        self._static = bool(cfg.get("mediapipe.static_image_mode", False))
        model_path = Path(str(cfg.get("mediapipe.model_path",
                                      "models/pose_landmarker_full.task")))
        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay model BlazePose: {model_path}. "
                "Tai pose_landmarker_*.task ve va dat vao models/ "
                "hoac chinh mediapipe.model_path trong config.")
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE if self._static else RunningMode.VIDEO,
            num_poses=int(cfg.get("mediapipe.num_poses", 1)),
            min_pose_detection_confidence=float(
                cfg.get("mediapipe.min_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(
                cfg.get("mediapipe.min_presence_confidence", 0.5)),
            min_tracking_confidence=float(
                cfg.get("mediapipe.min_tracking_confidence", 0.5)),
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._static:
            res = self._landmarker.detect(mp_image)
        else:
            self._timestamp_ms += 33  # tang don dieu, RunningMode.VIDEO yeu cau
            res = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        image_lm = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
        vis = np.array([_safe_vis(l) for l in lm], dtype=np.float32)
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            world_lm = np.array([[l.x, l.y, l.z] for l in wl], dtype=np.float32)
        else:
            world_lm = image_lm.copy()
        return PoseResult(world_landmarks=world_lm, image_landmarks=image_lm,
                          visibility=vis)

    def close(self) -> None:
        self._landmarker.close()


def _safe_vis(landmark) -> float:
    v = getattr(landmark, "visibility", None)
    if v is None:
        v = getattr(landmark, "presence", None)
    return float(v) if v is not None else 1.0
