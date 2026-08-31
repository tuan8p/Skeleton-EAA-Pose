"""Logger and VideoStats for independent pipeline detection.

Tracking metrics pure detection (no skeleton):
- ok_frames: frames with enough bbox >= conf_threshold
- fail_frames: frames without enough bbox after retry
- empty_frames: subset of fail_frames, YOLO returned 0 bbox
- retry_count: total retry count (max 2/frame)
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .config_manager import ConfigManager


class DetectionVideoStats:
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.t_start = time.time()
        self.t_end: float | None = None
        self.total_frames: int = 0          # total unique frames processed (per annotation)
        self.ok_frames: int = 0             # frame has enough n_expected bbox >= threshold
        self.fail_frames: list[int] = []    # frame_id fail sau retry
        self.empty_frames: list[int] = []   # subset fail: YOLO returns 0 bbox
        self.retry_count: int = 0
        self.detect_confs: list[float] = [] # max conf for each successful frame
        self.video_fps: float = 0.0
        self.chunk_stats: list[dict] = []   # [{chunk_idx, total, fail, ok}, ...]

    def finish(self) -> None:
        self.t_end = time.time()

    @property
    def elapsed(self) -> float:
        return (self.t_end or time.time()) - self.t_start

    @property
    def avg_fps(self) -> float:
        return self.total_frames / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def avg_detect_conf(self) -> float:
        return float(sum(self.detect_confs) / len(self.detect_confs)) \
            if self.detect_confs else 0.0

    @property
    def video_ok(self) -> bool:
        return len(self.fail_frames) == 0

    def summary_row(self) -> dict:
        return {
            "video_name": self.video_name,
            "total_frames": self.total_frames,
            "ok_frames": self.ok_frames,
            "fail_frames_count": len(self.fail_frames),
            "empty_frames_count": len(self.empty_frames),
            "video_ok": self.video_ok,
            "retry_count": self.retry_count,
            "avg_detect_conf": round(self.avg_detect_conf, 4),
            "fps": round(self.avg_fps, 2),
            "avg_time_per_frame_ms": round(self.elapsed / self.total_frames * 1000, 2)
            if self.total_frames else 0.0,
        }


class DetectionLogger:
    def __init__(self, cfg: ConfigManager, dataset: str,
                 batch_start: int, batch_end: int):
        detect_out = Path(cfg.get("detection.output_dir", "out/outputs_detect"))
        self.log_dir = detect_out / dataset.upper() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        level = getattr(
            logging, str(cfg.get("runtime.log_level", "INFO")).upper(), logging.INFO)
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(message)s")
        self._log = logging.getLogger("detection")
        self.batch_csv = self.log_dir / f"batch_summary_{batch_start}_{batch_end}.csv"

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def write_video_log(self, stats: DetectionVideoStats, cfg_dict: dict) -> None:
        row = stats.summary_row()
        log_file = self.log_dir / f"{stats.video_name}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"video: {stats.video_name}\n")
            f.write(f"start: {time.ctime(stats.t_start)}\n")
            f.write(f"end: {time.ctime(stats.t_end or time.time())}\n")
            f.write(f"config: {cfg_dict}\n")
            f.write(f"video_fps: {stats.video_fps}\n")
            f.write(f"total_frames: {row['total_frames']}\n")
            f.write(f"ok_frames: {row['ok_frames']}\n")
            f.write(f"fail_frames_count: {row['fail_frames_count']}\n")
            f.write(f"empty_frames_count: {row['empty_frames_count']} "
                    f"(YOLO tra 0 bbox)\n")
            f.write(f"video_ok: {row['video_ok']}\n")
            f.write(f"retry_count: {row['retry_count']}\n")
            f.write(f"avg_detect_conf: {row['avg_detect_conf']}\n")
            f.write(f"processing_fps: {row['fps']}\n")
            f.write(f"avg_time_per_frame_ms: {row['avg_time_per_frame_ms']}\n")
            if stats.fail_frames:
                f.write(f"fail_frames: {stats.fail_frames}\n")
            if stats.empty_frames:
                f.write(f"empty_frames: {stats.empty_frames}\n")
            for cs in stats.chunk_stats:
                f.write(
                    f"chunk_{cs['chunk_idx']:03d}: "
                    f"total={cs['total']}, fail={cs['fail']}, ok={cs['ok']}\n"
                )

    def append_batch_summary(self, stats: DetectionVideoStats) -> None:
        row = stats.summary_row()
        write_header = not self.batch_csv.exists()
        with open(self.batch_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
