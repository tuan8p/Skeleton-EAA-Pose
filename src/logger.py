"""Log chi tiet tung video + file thong ke batch (PLAN muc 7).

ok_frames: so frame OK SAU retry + noi suy (frame du joints, khong con NaN;
action tuong tac can du 2 person). error_frames: loi cung LUC EXTRACT
(no_detect/low_conf/pose_fail). bad_frames: frame KHONG ok sau noi suy.
video_ok = True khi khong co bad frame nao.
"""
from __future__ import annotations

import csv
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config_manager import ConfigManager


class VideoStats:
    def __init__(self, video_name: str):
        self.video_name = video_name
        self.t_start = time.time()
        self.t_end: float | None = None
        self.total_frames = 0
        self.ok_frames = 0       # frame du joints SAU retry + noi suy
        self.ok_persons = 0      # so person hop le LUC EXTRACT (frame 2 nguoi: toi da 2)
        self.error_frames: dict[str, list[int]] = defaultdict(list)  # loi luc extract
        self.bad_frames: list[int] = []   # frame con NaN sau noi suy
        # Thong ke theo YOLO detect:
        self.detect_total = 0            # so frame da chay YOLO
        self.detect_ok = 0               # so frame co it nhat 1 bbox person
        self.detect_confs: list[float] = []   # conf bbox cao nhat moi frame co detect
        self.fn_frames: list[int] = []   # False Negative: co nguoi bi sot (da retry/noi suy bbox)
        self.tn_frames: list[int] = []   # True Negative: thuc su khong co nguoi
        self.interpolated_frames: list[int] = []  # frame duoc noi suy (bbox hoac skeleton)
        self.retry_count = 0
        self.conf_values: list[float] = []   # mean conf cua cac frame thanh cong
        self.video_fps = 0.0

    @property
    def detect_fail(self) -> int:
        return self.detect_total - self.detect_ok

    @property
    def avg_detect_conf(self) -> float:
        return float(sum(self.detect_confs) / len(self.detect_confs)) \
            if self.detect_confs else 0.0

    def finish(self) -> None:
        self.t_end = time.time()

    @property
    def elapsed(self) -> float:
        return (self.t_end or time.time()) - self.t_start

    @property
    def avg_fps(self) -> float:
        return self.total_frames / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def avg_conf(self) -> float:
        return float(sum(self.conf_values) / len(self.conf_values)) if self.conf_values else 0.0

    @property
    def conf_mode(self) -> float:
        """Mode cua confidence (lam tron 2 chu so) trong cac frame thanh cong."""
        if not self.conf_values:
            return 0.0
        return float(Counter(round(c, 2) for c in self.conf_values).most_common(1)[0][0])

    @property
    def video_ok(self) -> bool:
        return len(self.bad_frames) == 0

    def summary_row(self) -> dict:
        return {
            "video_name": self.video_name,
            "total_frames": self.total_frames,
            "ok_frames": self.ok_frames,
            "ok_persons": self.ok_persons,
            "bad_frames_count": len(self.bad_frames),
            "video_ok": self.video_ok,
            "no_detect_count": len(self.error_frames.get("no_detect", [])),
            "low_conf_count": len(self.error_frames.get("low_conf", [])),
            "pose_fail_count": len(self.error_frames.get("pose_fail", [])),
            "retry_count": self.retry_count,
            "detect_ok_frames": self.detect_ok,
            "detect_fail_frames": self.detect_fail,
            "avg_detect_conf": round(self.avg_detect_conf, 4),
            "fn_count": len(self.fn_frames),
            "tn_count": len(self.tn_frames),
            "interpolated_count": len(self.interpolated_frames),
            "fps": round(self.avg_fps, 2),
            "avg_time_per_frame_ms": round(self.elapsed / self.total_frames * 1000, 2)
            if self.total_frames else 0.0,
            "avg_conf": round(self.avg_conf, 4),
            "conf_mode": round(self.conf_mode, 2),
        }


class PipelineLogger:
    def __init__(self, cfg: ConfigManager, batch_start: int, batch_end: int):
        self.log_dir = Path(cfg.get("paths.log_dir"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        level = getattr(logging, str(cfg.get("runtime.log_level", "INFO")).upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
        self._log = logging.getLogger("pipeline")
        self.batch_csv = self.log_dir / f"batch_summary_{batch_start}_{batch_end}.csv"

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def write_video_log(self, stats: VideoStats, cfg_dict: dict) -> None:
        row = stats.summary_row()
        log_file = self.log_dir / f"{stats.video_name}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"video: {stats.video_name}\n")
            f.write(f"start: {time.ctime(stats.t_start)}\n")
            f.write(f"end: {time.ctime(stats.t_end or time.time())}\n")
            f.write(f"config: {cfg_dict}\n")
            f.write(f"video_fps: {stats.video_fps}\n")
            f.write(f"total_frames: {row['total_frames']}\n")
            f.write(f"ok_frames: {row['ok_frames']} (sau retry + noi suy)\n")
            f.write(f"ok_persons: {row['ok_persons']} (luc extract)\n")
            f.write(f"bad_frames_count: {row['bad_frames_count']}\n")
            f.write(f"video_ok: {row['video_ok']}\n")
            f.write(f"detect_ok_frames (YOLO co bbox): {row['detect_ok_frames']}\n")
            f.write(f"detect_fail_frames (YOLO khong bbox): {row['detect_fail_frames']}\n")
            f.write(f"avg_detect_conf: {row['avg_detect_conf']}\n")
            if stats.fn_frames:
                f.write(f"fn_frames (co nguoi bi sot, da retry/noi suy bbox): "
                        f"{stats.fn_frames}\n")
            if stats.tn_frames:
                f.write(f"tn_frames (thuc su khong nguoi, skeleton=0): "
                        f"{stats.tn_frames}\n")
            if stats.interpolated_frames:
                f.write(f"interpolated_frames: {stats.interpolated_frames}\n")
            if stats.bad_frames:
                f.write(f"bad_frames: {stats.bad_frames}\n")
            for err_type, frames in sorted(stats.error_frames.items()):
                f.write(f"{err_type}_at_extract: {len(frames)} frames -> {frames}\n")
            f.write(f"retry_count: {row['retry_count']}\n")
            f.write(f"processing_fps: {row['fps']}\n")
            f.write(f"avg_time_per_frame_ms: {row['avg_time_per_frame_ms']}\n")
            f.write(f"avg_conf: {row['avg_conf']}\n")
            f.write(f"conf_mode: {row['conf_mode']}\n")
            f.write("vram_used: n/a (CPU-only)\n")

    def append_batch_summary(self, stats: VideoStats) -> None:
        row = stats.summary_row()
        write_header = not self.batch_csv.exists()
        with open(self.batch_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
