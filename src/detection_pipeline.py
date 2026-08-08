"""Pipeline detection độc lập (không BlazePose / skeleton).

Luồng per-video:
1. Đọc annotation → list[ActionSegment] (đã lọc theo event_mapping TSU / PKU)
2. Phát hiện video có action tương tác → video_is_interactive → n_expected = 2
3. Chia chunk (≤ chunk.max_actions_per_chunk = 5 segment/chunk)
4. Per chunk (skip nếu progress đã done VÀ temp jsonl tồn tại):
   a. Load frame_cache từ chunk trước (overlap frames) nếu resume
   b. Reset tracker
   c. _detect_chunk(): union intervals → detect frame (cache-first) → retry
   d. Ghi temp jsonl → save_chunk_visualize (nếu chunk fail) → mark_chunk_done
5. Cuối video: merge_to_video_jsonl → write_video_log → mark_video_done

Retry strategy (khi frame không đủ n_expected bbox):
  Attempt 0: YOLO detect gốc
  Retry 1:   YOLO với imgsz=imgsz_retry (1280) + augment=tta
  Retry 2:   Preprocess ảnh (brightness + CLAHE nếu bật) → YOLO gốc

Frame cache: {fid: list[PersonBox]} sống xuyên suốt video.
  - Overlap frames giữa chunk A-1 và chunk A: served từ cache (không detect lại)
  - Resume: load lại từ temp jsonl của chunk A-1 để populate cache

Dedup stats: mỗi frame_id chỉ count 1 lần vào total_frames/ok/fail
  (tránh double-count khi frame xuất hiện trong union của 2 chunk khác nhau)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .action_segment import ActionSegment
from .config_manager import ConfigManager
from .detection_annotator import get_detection_annotator
from .detection_logger import DetectionLogger, DetectionVideoStats
from .detection_saver import DetectionSaver
from .progress_manager import ProgressManager
from .tracker import get_person_tracker
from .yolo_detector import PersonBox, YOLODetector

NUM_PERSONS = 2
# PKU interactive action IDs: cần 2 người
_DEFAULT_INTERACTIVE_IDS = {12, 14, 16, 18, 21, 24, 26, 27}


# ─────────────────────────── helpers ───────────────────────────

def _build_chunks(segments: list[ActionSegment],
                  max_per_chunk: int) -> list[list[ActionSegment]]:
    segs = sorted(segments, key=lambda s: s.start_frame)
    return [segs[i: i + max_per_chunk] for i in range(0, len(segs), max_per_chunk)]


def _union_intervals(segs: list[ActionSegment]) -> list[tuple[int, int, list[int]]]:
    """Hợp nhất [start, end] chồng lặp → list (start, end, [action_ids]).

    Mỗi frame chỉ xuất hiện 1 lần trong kết quả; vùng giao mang nhiều action_id.
    """
    events: list[tuple[int, int, int]] = []
    for s in segs:
        events.append((s.start_frame, 1, s.action_id))
        events.append((s.end_frame + 1, -1, s.action_id))
    events.sort(key=lambda e: (e[0], -e[1]))
    intervals: list[tuple[int, int, list[int]]] = []
    active: list[int] = []
    prev: int | None = None
    for pos, delta, aid in events:
        if prev is not None and pos > prev and active:
            intervals.append((prev, pos - 1, sorted(active)))
        if delta == 1:
            active.append(aid)
        else:
            if aid in active:
                active.remove(aid)
        prev = pos
    return intervals


def _make_row(fid: int, fps: float, action_ids: list[int],
              chunk_idx: int, bboxes: list,
              ok: bool, fail_reason: str | None) -> dict:
    """Tạo 1 dòng metadata frame cho output jsonl."""
    aid_out = action_ids[0] if len(action_ids) == 1 else action_ids
    return {
        "frame_id": int(fid),
        "timestamp": round(fid / fps, 4),
        "action_id": aid_out,
        "chunk_index": chunk_idx,
        "bboxes": bboxes,       # [[x1,y1,x2,y2,conf], ...]  tọa độ frame gốc
        "n_persons": len(bboxes),
        "ok": ok,
        "fail_reason": fail_reason,  # "no_detect" | "low_conf" | "missing_person" | null
    }


def _apply_retry_preproc(frame: np.ndarray,
                         brightness_delta: int,
                         clahe_on: bool) -> np.ndarray:
    """Áp dụng tiền xử lý nhẹ cho retry: brightness + CLAHE (tùy chọn)."""
    out = frame.copy()
    if brightness_delta != 0:
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=brightness_delta)
    if clahe_on:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


# ─────────────────────────── pipeline ───────────────────────────

class DetectionPipeline:
    def __init__(self, config_path: str | Path | None = None,
                 cfg: ConfigManager | None = None,
                 yolo_detector=None):
        self.cfg = cfg or ConfigManager(config_path)
        self.dataset = str(self.cfg.get("dataset", "PKU")).upper()
        self.annotator = get_detection_annotator(self.cfg)
        self.yolo: YOLODetector = (
            yolo_detector if yolo_detector is not None else YOLODetector(self.cfg))

        # Thresholds
        self.conf_threshold = float(self.cfg.get("yolo.conf_threshold", 0.5))
        self.interactive_ids: set[int] = set(
            self.cfg.get("pku.interactive_action_ids",
                         list(_DEFAULT_INTERACTIVE_IDS)))

        # Chunking
        self.max_per_chunk = int(self.cfg.get("chunk.max_actions_per_chunk", 5))
        self.max_segments = self.cfg.get("runtime.max_segments_per_video", None)
        self.max_chunks = self.cfg.get("runtime.max_chunks_per_video", None)

        # Retry config
        self.max_retries = int(self.cfg.get("detection.retry.max_retries", 2))
        self.imgsz_retry = int(self.cfg.get("detection.retry.imgsz_retry", 1280))
        self.tta = bool(self.cfg.get("detection.retry.tta", False))
        self.brightness_delta = int(self.cfg.get("detection.retry.brightness_delta", 30))
        self.clahe_on_retry = bool(self.cfg.get("detection.retry.clahe_on_retry", False))

        # Output infra
        detect_out = Path(self.cfg.get("detection.output_dir", "out/outputs_detect"))
        ds_dir = detect_out / self.dataset
        progress_path = ds_dir / "bboxes" / "progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress = ProgressManager(progress_path)
        self.saver = DetectionSaver(self.cfg, self.dataset)

    # ──────────── public API ────────────

    def run_batch(self, start: int = 0, end: int | None = None) -> None:
        videos = self.annotator.list_videos()
        end = len(videos) if end is None else min(end, len(videos))
        logger = DetectionLogger(self.cfg, self.dataset, start, end)
        for video_name in videos[start:end]:
            try:
                stats = self.process_video(video_name, logger)
                logger.write_video_log(stats, self.cfg.as_dict())
                logger.append_batch_summary(stats)
                logger.info(
                    f"Xong {video_name}: ok={stats.ok_frames}/{stats.total_frames}, "
                    f"fail={len(stats.fail_frames)}, video_ok={stats.video_ok}")
            except Exception as exc:
                logger.warning(f"Lỗi video {video_name}: {exc}")
                import traceback
                logger.warning(traceback.format_exc())

    def process_video(self, video_name: str,
                      logger: DetectionLogger | None = None) -> DetectionVideoStats:
        stats = DetectionVideoStats(video_name)
        segments = self.annotator.read(video_name)
        if not segments:
            stats.finish()
            return stats
        if self.max_segments:
            segments = segments[: int(self.max_segments)]

        video_path = segments[0].video_path

        # Lấy FPS
        cap_probe = cv2.VideoCapture(video_path)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        cap_probe.release()
        stats.video_fps = fps

        # Phát hiện video có action tương tác không (PKU only)
        video_is_interactive = (
            self.dataset == "PKU"
            and any(s.action_id in self.interactive_ids for s in segments)
        )
        n_expected = NUM_PERSONS if video_is_interactive else 1

        # Tracker instance (dùng chung cho cả video, reset giữa các chunk)
        tracker = get_person_tracker(self.cfg) if video_is_interactive else None

        # Frame cache: {fid → list[PersonBox]}, sống xuyên suốt video
        frame_cache: dict[int, list[PersonBox]] = {}
        # Set frame_id đã count vào stats (tránh double-count overlap)
        counted_frames: set[int] = set()

        chunks = _build_chunks(segments, self.max_per_chunk)
        if self.max_chunks is not None:
            chunks = chunks[: int(self.max_chunks)]

        for chunk_idx, chunk_segs in enumerate(chunks):
            # ── Resume check ──
            # Quét folder chunks: nếu file temp jsonl của chunk đã tồn tại thì coi như done
            if self.saver.chunk_exists(video_name, chunk_idx):
                # Chunk đã done → nạp lại vào frame_cache cho chunk sau
                done_rows = self.saver.load_chunk_rows(video_name, chunk_idx)
                frame_cache.update(DetectionSaver.rows_to_frame_cache(done_rows))
                # Cũng update counted_frames để tránh double-count
                for row in done_rows:
                    fid = row["frame_id"]
                    if fid not in counted_frames:
                        counted_frames.add(fid)
                        stats.total_frames += 1
                        if row.get("ok", False):
                            stats.ok_frames += 1
                            bboxes = row.get("bboxes", []) or []
                            if bboxes:
                                stats.detect_confs.append(
                                    max(float(b[4]) for b in bboxes))
                        else:
                            stats.fail_frames.append(fid)
                            if not row.get("bboxes"):
                                stats.empty_frames.append(fid)
                if logger:
                    logger.info(
                        f"{video_name} chunk {chunk_idx}: đã done, bỏ qua")
                continue

            # ── Reset tracker đầu mỗi chunk ──
            if tracker is not None:
                tracker.reset()

            # ── Detect chunk ──
            chunk_rows, chunk_ok = self._detect_chunk(
                video_path, chunk_segs, chunk_idx,
                frame_cache, counted_frames,
                tracker, n_expected, fps, stats)

            # ── Ghi output ──
            self.saver.save_chunk_rows(video_name, chunk_idx, chunk_rows)

            if not chunk_ok:
                seg_start = min(s.start_frame for s in chunk_segs)
                seg_end = max(s.end_frame for s in chunk_segs)
                self.saver.save_chunk_visualize(
                    video_path, video_name, chunk_idx,
                    chunk_rows, seg_start, seg_end, self.conf_threshold)

            # ── Chunk stats ──
            chunk_fail_count = sum(1 for r in chunk_rows if not r.get("ok", True))
            stats.chunk_stats.append({
                "chunk_idx": chunk_idx,
                "total": len(chunk_rows),
                "fail": chunk_fail_count,
                "ok": chunk_ok,
            })
            self.progress.mark_chunk_done(video_name, chunk_idx, len(chunks))

        # ── Cuối video: merge → video jsonl ──
        self.saver.merge_to_video_jsonl(video_name, len(chunks))
        stats.finish()
        self.progress.mark_video_done(video_name)
        return stats

    # ──────────── detect chunk ────────────

    def _detect_chunk(self, video_path: str,
                      chunk_segs: list[ActionSegment],
                      chunk_idx: int,
                      frame_cache: dict[int, list[PersonBox]],
                      counted_frames: set[int],
                      tracker, n_expected: int,
                      fps: float,
                      stats: DetectionVideoStats) -> tuple[list[dict], bool]:
        """Detect tất cả frame trong chunk. Trả về (rows, chunk_ok)."""
        intervals = _union_intervals(chunk_segs)
        rows: list[dict] = []
        chunk_ok = True

        cap = cv2.VideoCapture(video_path)
        try:
            for (start, end, action_ids) in intervals:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
                for fid in range(start, end + 1):
                    read_ok, frame = cap.read()
                    if not read_ok:
                        # Không đọc được frame → fail cứng
                        self._count_frame(
                            fid, False, [], counted_frames, stats)
                        row = _make_row(fid, fps, action_ids, chunk_idx,
                                        [], False, "no_detect")
                        rows.append(row)
                        chunk_ok = False
                        continue

                    from_cache = fid in frame_cache
                    if from_cache:
                        boxes = frame_cache[fid]
                    else:
                        boxes = self._detect_with_retry(
                            frame, tracker, n_expected, stats)
                        frame_cache[fid] = boxes

                    valid = [b for b in boxes
                             if b.confidence >= self.conf_threshold]
                    frame_ok = (len(valid) == n_expected)

                    if not from_cache:
                        # Count stats chỉ lần đầu detect (không re-count cache hit)
                        self._count_frame(
                            fid, frame_ok, valid, counted_frames, stats)

                    if not frame_ok:
                        chunk_ok = False

                    bbox_out = [[*b.xyxy, round(b.confidence, 4)]
                                for b in boxes] if boxes else []
                    if frame_ok:
                        fail_reason = None
                    elif len(boxes) == 0:
                        fail_reason = "no_detect"
                    elif len(valid) == 0:
                        fail_reason = "low_conf"
                    else:
                        fail_reason = "missing_person"

                    row = _make_row(fid, fps, action_ids, chunk_idx,
                                    bbox_out, frame_ok, fail_reason)
                    rows.append(row)
        finally:
            cap.release()

        return rows, chunk_ok

    @staticmethod
    def _count_frame(fid: int, frame_ok: bool,
                     valid_boxes: list[PersonBox],
                     counted_frames: set[int],
                     stats: DetectionVideoStats) -> None:
        """Update stats cho frame, chỉ 1 lần (skip nếu đã counted)."""
        if fid in counted_frames:
            return
        counted_frames.add(fid)
        stats.total_frames += 1
        if frame_ok:
            stats.ok_frames += 1
            if valid_boxes:
                stats.detect_confs.append(
                    max(b.confidence for b in valid_boxes))
        else:
            stats.fail_frames.append(fid)
            if not valid_boxes:
                stats.empty_frames.append(fid)

    # ──────────── retry detect ────────────

    def _detect_with_retry(self, frame: np.ndarray,
                           tracker, n_expected: int,
                           stats: DetectionVideoStats) -> list[PersonBox]:
        """YOLO detect với retry. Trả về best boxes.

        Attempt 0 (original): imgsz mặc định (YOLO tự quyết)
        Retry 1: imgsz=imgsz_retry (1280) + augment=tta
        Retry 2: preprocess ảnh (brightness / CLAHE) + imgsz mặc định
        """
        best_boxes: list[PersonBox] = []
        best_valid_count = 0
        best_max_conf = -1.0

        # Attempt 0 — original
        attempt_boxes = self.yolo.detect(frame)
        best_boxes, best_valid_count, best_max_conf = self._pick_best(
            attempt_boxes, best_boxes, best_valid_count, best_max_conf)

        if best_valid_count >= n_expected:
            return self._apply_tracker(best_boxes, frame, tracker, n_expected)

        # Retry 1 — imgsz lớn hơn + TTA
        if self.max_retries >= 1:
            retry_boxes1 = self.yolo.detect(
                frame, imgsz=self.imgsz_retry, augment=self.tta)
            best_boxes, best_valid_count, best_max_conf = self._pick_best(
                retry_boxes1, best_boxes, best_valid_count, best_max_conf)
            stats.retry_count += 1
            if best_valid_count >= n_expected:
                return self._apply_tracker(best_boxes, frame, tracker, n_expected)

        # Retry 2 — preprocess ảnh
        if self.max_retries >= 2:
            preproc = _apply_retry_preproc(
                frame, self.brightness_delta, self.clahe_on_retry)
            retry_boxes2 = self.yolo.detect(preproc)
            best_boxes, best_valid_count, best_max_conf = self._pick_best(
                retry_boxes2, best_boxes, best_valid_count, best_max_conf)
            stats.retry_count += 1

        return self._apply_tracker(best_boxes, frame, tracker, n_expected)

    def _pick_best(self, new_boxes: list[PersonBox],
                   cur_best: list[PersonBox],
                   cur_count: int,
                   cur_conf: float) -> tuple[list[PersonBox], int, float]:
        """So sánh và giữ kết quả tốt hơn (nhiều valid box hơn, hoặc conf cao hơn)."""
        valid_new = [b for b in new_boxes if b.confidence >= self.conf_threshold]
        max_conf_new = max((b.confidence for b in valid_new), default=0.0)
        # Ưu tiên: số valid box nhiều hơn; nếu bằng → conf cao hơn
        if (len(valid_new) > cur_count
                or (len(valid_new) == cur_count and max_conf_new > cur_conf)):
            return new_boxes, len(valid_new), max_conf_new
        return cur_best, cur_count, cur_conf

    def _apply_tracker(self, boxes: list[PersonBox],
                       frame: np.ndarray,
                       tracker, n_expected: int) -> list[PersonBox]:
        """Áp dụng tracker (chỉ khi n_expected == 2). Trả về sorted/sliced boxes."""
        if tracker is not None and boxes:
            return tracker.update(boxes, frame)[:NUM_PERSONS]
        return boxes[:n_expected]
