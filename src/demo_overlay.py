"""Demo overlay: doc .npy + .jsonl, ve skeleton len video goc, xuat video moi.

Uu tien dung landmarks_2d (normalized theo frame goc) trong jsonl; neu khong co
thi dung 2 cot dau cua npy (chi dung duoc khi npy luu toa do image normalized).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .mapper import NTU_BONES, NUM_NTU_JOINTS


def _iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def create_overlay_video(video_path: str, skeleton_file: str, metadata_file: str,
                         output_video: str, joint_color=(0, 255, 0),
                         line_color=(0, 0, 255)) -> None:
    skeleton = np.load(skeleton_file)  # (T, 2, 25, 3)
    meta_by_frame = {row["frame_id"]: row for row in _iter_jsonl(metadata_file)}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Khong mo duoc video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fid = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1  # annotation 1-based
            row = meta_by_frame.get(fid)
            if row is not None:
                lm2d = row.get("landmarks_2d")
                if lm2d is not None:
                    _draw(frame, np.asarray(lm2d, dtype=np.float32)[..., :2], (w, h),
                          joint_color, line_color)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def _draw(frame: np.ndarray, lm2d: np.ndarray, wh: tuple[int, int],
          joint_color, line_color) -> None:
    w, h = wh
    persons = lm2d if lm2d.ndim == 3 else lm2d[None, ...]
    for person in persons[:2]:
        pts = person
        valid = ~np.isnan(pts).any(axis=1)
        px = np.full((pts.shape[0], 2), -1, dtype=int)
        px[valid, 0] = (pts[valid, 0] * w).astype(int)
        px[valid, 1] = (pts[valid, 1] * h).astype(int)
        for a, b in NTU_BONES:
            if a < len(px) and b < len(px) and valid[a] and valid[b]:
                cv2.line(frame, tuple(px[a]), tuple(px[b]), line_color, 2)
        for j in range(min(NUM_NTU_JOINTS, len(px))):
            if valid[j]:
                cv2.circle(frame, tuple(px[j]), 3, joint_color, -1)
