"""Lưu output pipeline detection: per-chunk temp jsonl, visualize video, video jsonl.

Layout output:
    {detection.output_dir}/{DATASET}/
        bboxes/
            jsonl/<video>.jsonl          # final per-video (sau khi merge chunks)
            chunks/<video>_chunk_NNN.jsonl  # temp per-chunk (dùng cho resume + merge)
            progress.json
        failed_frames/<video>/chunk_NNN/
            <video>_chunk_NNN.mp4        # video visualize chunk (chỉ khi chunk fail)
            frame_XXXXXX.jpg             # ảnh tĩnh từng frame fail
        logs/                            # xử lý bởi DetectionLogger

Bbox trong output: tọa độ frame GỐC (YOLO Ultralytics tự trả về coord gốc).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config_manager import ConfigManager
from .yolo_detector import PersonBox

_OK_COLOR = (0, 200, 0)      # xanh lá — bbox ok
_FAIL_COLOR = (0, 0, 220)    # đỏ — bbox fail / no detect
_LOW_COLOR = (0, 140, 255)   # cam — bbox low conf


class DetectionSaver:
    def __init__(self, cfg: ConfigManager, dataset: str):
        detect_out = Path(cfg.get("detection.output_dir", "out/outputs_detect"))
        ds = dataset.upper()
        self.bboxes_dir = detect_out / ds / "bboxes"
        self.jsonl_dir = self.bboxes_dir / "jsonl"
        self.chunks_tmp_dir = self.bboxes_dir / "chunks"  # temp, không expose ra user
        self.failed_dir = detect_out / ds / "failed_frames"
        for d in (self.jsonl_dir, self.chunks_tmp_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- per-chunk temp I/O ----------

    def chunk_rows_path(self, video_name: str, chunk_idx: int) -> Path:
        return self.chunks_tmp_dir / f"{video_name}_chunk_{chunk_idx:03d}.jsonl"

    def chunk_exists(self, video_name: str, chunk_idx: int) -> bool:
        return self.chunk_rows_path(video_name, chunk_idx).exists()

    def save_chunk_rows(self, video_name: str, chunk_idx: int,
                        rows: list[dict]) -> None:
        """Ghi kết quả chunk ra temp jsonl (gọi sau khi chunk xong)."""
        path = self.chunk_rows_path(video_name, chunk_idx)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_chunk_rows(self, video_name: str, chunk_idx: int) -> list[dict]:
        """Đọc lại temp jsonl của chunk (dùng khi resume để lấy cache)."""
        path = self.chunk_rows_path(video_name, chunk_idx)
        if not path.exists():
            return []
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def rows_to_frame_cache(rows: list[dict]) -> dict[int, list[PersonBox]]:
        """Reconstruct frame_cache từ saved rows (dùng khi resume)."""
        cache: dict[int, list[PersonBox]] = {}
        for row in rows:
            fid = row["frame_id"]
            boxes: list[PersonBox] = []
            for b in row.get("bboxes", []) or []:
                x1, y1, x2, y2, conf = int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])
                boxes.append(PersonBox(xyxy=(x1, y1, x2, y2), confidence=conf))
            cache[fid] = boxes
        return cache

    # ---------- final video jsonl ----------

    def merge_to_video_jsonl(self, video_name: str, n_chunks: int) -> None:
        """Merge tất cả chunk temp jsonls → 1 video jsonl (sorted by frame_id, dedup).

        Dedup: frame_id xuất hiện trong nhiều chunk (overlap) → giữ row từ chunk đầu tiên.
        """
        seen_fids: set[int] = set()
        all_rows: list[dict] = []
        for ci in range(n_chunks):
            for row in self.load_chunk_rows(video_name, ci):
                fid = row["frame_id"]
                if fid not in seen_fids:
                    seen_fids.add(fid)
                    all_rows.append(row)
        all_rows.sort(key=lambda r: r["frame_id"])
        out = self.jsonl_dir / f"{video_name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- chunk visualize ----------

    def save_chunk_visualize(self, video_path: str, video_name: str,
                             chunk_idx: int, chunk_rows: list[dict],
                             seg_start: int, seg_end: int,
                             conf_threshold: float) -> None:
        """Sinh video visualize cho chunk và ảnh tĩnh các frame fail.

        Video bao gồm tất cả frame [seg_start, seg_end]:
        - Frame trong annotation: vẽ bbox + conf + frame_id + action_id
        - Frame không trong annotation: ghi nguyên frame gốc
        Frame fail → lưu thêm frame_XXXXXX.jpg riêng.
        Chỉ gọi khi chunk_ok = False.
        """
        viz_dir = self.failed_dir / video_name / f"chunk_{chunk_idx:03d}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_mp4 = str(viz_dir / f"{video_name}_chunk_{chunk_idx:03d}.mp4")
        writer = cv2.VideoWriter(
            out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (w, h))

        rows_by_fid: dict[int, dict] = {r["frame_id"]: r for r in chunk_rows}
        fail_fids: set[int] = {r["frame_id"] for r in chunk_rows
                               if not r.get("ok", True)}

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start - 1)
            for fid in range(seg_start, seg_end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                row = rows_by_fid.get(fid)
                if row is not None:
                    out_f = frame.copy()
                    _draw_detection_frame(out_f, row, conf_threshold)
                    if fid in fail_fids:
                        cv2.imwrite(str(viz_dir / f"frame_{fid:06d}.jpg"), out_f)
                    writer.write(out_f)
                else:
                    # Frame không thuộc annotation range nào trong chunk
                    writer.write(frame)
        finally:
            cap.release()
            writer.release()


# ---------- drawing helpers ----------

def _draw_detection_frame(frame: np.ndarray, row: dict,
                          conf_threshold: float) -> None:
    """Vẽ thông tin detection lên 1 frame (in-place)."""
    fid = row.get("frame_id", "?")
    aid = row.get("action_id", "?")
    ok = row.get("ok", True)
    fail_reason = row.get("fail_reason")

    # Header text: frame_id + action_id + status
    status = "OK" if ok else (fail_reason or "FAIL")
    header = f"F:{fid} A:{aid} [{status}]"
    hdr_color = _OK_COLOR if ok else _FAIL_COLOR
    cv2.putText(frame, header, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hdr_color, 2)

    # Vẽ bbox
    for b in row.get("bboxes", []) or []:
        x1, y1, x2, y2, conf = (
            int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4]))
        bc = _OK_COLOR if conf >= conf_threshold else _LOW_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(4, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bc, 1)

    # Nếu fail và không có bbox: chú thích thêm
    if not ok and not row.get("bboxes"):
        cv2.putText(frame, "NO BBOX", (6, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FAIL_COLOR, 2)
