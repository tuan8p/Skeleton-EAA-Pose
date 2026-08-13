"""Luu skeleton .npy + metadata .jsonl + anh frame loi."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config_manager import ConfigManager
from .validator import ERR_LOW_CONF


class SkeletonSaver:
    """Layout output:
        output_dir/chunks/npy/<video>_chunk_NNN.npy
        output_dir/chunks/jsonl/<video>_chunk_NNN.jsonl
        output_dir/actions/npy/<video>_NNN.npy      (tu split_actions)
        output_dir/actions/jsonl/<video>_NNN.jsonl
    """

    def __init__(self, cfg: ConfigManager):
        self.output_dir = Path(cfg.get("paths.output_dir"))
        self.csv_dir = self.output_dir / "csv"
        self.failed_dir = self.output_dir / "failed_frames"
        for d in (self.csv_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_video(self, video_name: str, skeleton: np.ndarray, metadata_rows: list[dict]) -> None:
        # Convert NaN to 0.0 as requested for failed/empty cases
        skeleton = np.nan_to_num(skeleton, nan=0.0)
        
        import csv
        csv_path = self.csv_dir / f"{video_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["frame_id"]
            for p in (1, 2):
                for j in range(1, 26):
                    header.extend([f"p{p}_x{j}", f"p{p}_y{j}", f"p{p}_z{j}"])
            writer.writerow(header)
            for t, row_meta in enumerate(metadata_rows):
                row_data = [row_meta["frame_id"]]
                flat = skeleton[t].flatten().round(4).tolist()
                row_data.extend(flat)
                writer.writerow(row_data)

    def save_failed_frame(self, video_name: str, frame_id: int, error_type: str,
                          frame: np.ndarray, draw_boxes: list | None = None) -> None:
        sub = self.failed_dir / video_name
        sub.mkdir(parents=True, exist_ok=True)
        base = sub / f"{video_name}_frame_{frame_id}_{error_type}.jpg"
        cv2.imwrite(str(base), frame)
        if error_type == ERR_LOW_CONF:
            vis = frame.copy()
            for box in draw_boxes or []:
                x1, y1, x2, y2 = box.xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(sub / f"{video_name}_frame_{frame_id}_d.jpg"), vis)
