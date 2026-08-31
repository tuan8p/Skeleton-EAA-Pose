"""Pipeline detection independent (no BlazePose/skeleton).

Per-video stream:
1. Read annotation → list[ActionSegment] (filtered by event_mapping TSU / PKU)
2. Detect videos with interactive actions → video_is_interactive → n_expected = 2
3. Chia chunk (≤ chunk.max_actions_per_chunk = 5 segment/chunk)
4. Per chunk (skip if progress is done AND temp jsonl exists):
   a. Load frame_cache from previous chunk (overlap frames) if resume
   b. Reset tracker
   c. _detect_chunk(): union intervals → detect frame (cache-first) → retry
   d. Write temp jsonl → save_chunk_visualize (if chunk fails) → mark_chunk_done
5. End of video: merge_to_video_jsonl -> write_video_log -> mark_video_done

Retry strategy (when frame does not have enough n_expected bbox):
  Attempt 0: Original YOLO detect
  Retry 1: YOLO with imgsz=imgsz_retry (1280) + augment=tta
  Retry 2: Preprocess image (brightness + CLAHE if enabled) → Original YOLO

Frame cache: {fid: list[PersonBox]} lives throughout the video.
  - Overlap frames between chunk A-1 and chunk A: served from cache (not detected again)
  - Resume: reload from temp jsonl of chunk A-1 to populate cache

Dedup stats: each frame_id only counts once in total_frames/ok/fail
  (avoid double-count when frame appears in union of 2 different chunks)
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
# PKU interactive action IDs: requires 2 people
_DEFAULT_INTERACTIVE_IDS = {12, 14, 16, 18, 21, 24, 26, 27}


# ─────────────────────────── helpers ───────────────────────────

def _build_chunks(segments: list[ActionSegment],
                  max_per_chunk: int) -> list[list[ActionSegment]]:
    segs = sorted(segments, key=lambda s: s.start_frame)
    return [segs[i: i + max_per_chunk] for i in range(0, len(segs), max_per_chunk)]


def _union_intervals(segs: list[ActionSegment]) -> list[tuple[int, int, list[int]]]:
    """Merge [start, end] duplicates → list (start, end, [action_ids]).

    Each frame appears only once in the results; intersection area carries multiple action_id.
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
    """Create 1 line of metadata frame for jsonl output."""
    aid_out = action_ids[0] if len(action_ids) == 1 else action_ids
    return {
        "frame_id": int(fid),
        "timestamp": round(fid / fps, 4),
        "action_id": aid_out,
        "chunk_index": chunk_idx,
        "bboxes": bboxes,       # [[x1,y1,x2,y2,conf], ...] original frame coordinates
        "n_persons": len(bboxes),
        "ok": ok,
        "fail_reason": fail_reason,  # "no_detect" | "low_conf" | "missing_person" | null
    }


def _apply_retry_preproc(frame: np.ndarray,
                         brightness_delta: int,
                         clahe_on: bool) -> np.ndarray:
    """Apply light preprocessing to retry: brightness + CLAHE (optional)."""
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
        
        self.num_workers = int(self.cfg.get("runtime.num_workers", 1))
        import threading
        self.thread_local = threading.local()
        
        if self.num_workers <= 1:
            self._shared_yolo: YOLODetector | None = (
                yolo_detector if yolo_detector is not None else YOLODetector(self.cfg))
        else:
            self._shared_yolo = None

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

    def _get_yolo(self):
        if self._shared_yolo is not None:
            return self._shared_yolo
        if not hasattr(self.thread_local, 'yolo'):
            self.thread_local.yolo = YOLODetector(self.cfg)
        return self.thread_local.yolo

    # ──────────── public API ────────────

    def run_batch(self, start: int = 0, end: int | None = None) -> None:
        from tqdm import tqdm
        videos = self.annotator.list_videos()
        end = len(videos) if end is None else min(end, len(videos))
        logger = DetectionLogger(self.cfg, self.dataset, start, end)
        
        video_list = videos[start:end]
        
        def _process(video_name):
            try:
                stats = self.process_video(video_name, logger, disable_pbar=(self.num_workers > 1))
                logger.write_video_log(stats, self.cfg.as_dict())
                logger.append_batch_summary(stats)
                msg = (f"Xong {video_name}: ok={stats.ok_frames}/{stats.total_frames}, "
                       f"fail={len(stats.fail_frames)}, video_ok={stats.video_ok}")
                logger.info(msg)
                tqdm.write(msg)
            except Exception as exc:
                msg_err = f"Video error {video_name}: {exc}"
                logger.warning(msg_err)
                tqdm.write(msg_err)
                import traceback
                logger.warning(traceback.format_exc())

        if self.num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_process, vn): vn for vn in video_list}
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Videos (Parallel)", position=0):
                    pass
        else:
            for video_name in tqdm(video_list, desc="Videos", position=0):
                _process(video_name)

    def process_video(self, video_name: str,
                      logger: DetectionLogger | None = None,
                      disable_pbar: bool = False) -> DetectionVideoStats:
        stats = DetectionVideoStats(video_name)
        segments = self.annotator.read(video_name)
        if not segments:
            stats.finish()
            return stats
        if self.max_segments:
            segments = segments[: int(self.max_segments)]

        video_path = segments[0].video_path

        # Get FPS
        cap_probe = cv2.VideoCapture(video_path)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
        cap_probe.release()
        stats.video_fps = fps

        # Detect whether videos have interactive actions (PKU only)
        video_is_interactive = (
            self.dataset == "PKU"
            and any(s.action_id in self.interactive_ids for s in segments)
        )
        n_expected = NUM_PERSONS if video_is_interactive else 1

        # Tracker instance (common for the whole video, reset between chunks)
        tracker = get_person_tracker(self.cfg) if video_is_interactive else None

        # Frame cache: {fid → list[PersonBox]}, live throughout the video
        frame_cache: dict[int, list[PersonBox]] = {}
        # Set count frame_id to stats (avoid double-count overlap)
        counted_frames: set[int] = set()

        chunks = _build_chunks(segments, self.max_per_chunk)
        if self.max_chunks is not None:
            chunks = chunks[: int(self.max_chunks)]

        from tqdm import tqdm
        chunk_iterable = tqdm(enumerate(chunks), total=len(chunks), desc=f"Chunks ({video_name})", position=1, leave=False, disable=disable_pbar)

        for chunk_idx, chunk_segs in chunk_iterable:
            # ── Resume check ──
            # Scan the chunks folder: if the temp jsonl file of the chunk already exists, it is considered done
            if self.saver.chunk_exists(video_name, chunk_idx):
                # Chunk done → reloaded into frame_cache for the next chunk
                done_rows = self.saver.load_chunk_rows(video_name, chunk_idx)
                frame_cache.update(DetectionSaver.rows_to_frame_cache(done_rows))
                # Also update counted_frames to avoid double-count
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
                        f"{video_name} chunk {chunk_idx}: done, skipped")
                continue

            # ── Reset tracker at the beginning of each chunk ──
            if tracker is not None:
                tracker.reset()

            # ── Detect chunk ──
            chunk_rows, chunk_ok = self._detect_chunk(
                video_path, chunk_segs, chunk_idx,
                frame_cache, counted_frames,
                tracker, fps, stats)

            # ── Write output ──
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

        # ── End of video: merge → video jsonl ──
        self.saver.merge_to_video_jsonl(video_name, len(chunks))
        stats.finish()
        self.progress.mark_video_done(video_name)
        return stats

    # ──────────── detect chunk ────────────

    def _detect_chunk(self, video_path: str, chunk_segs: list[ActionSegment],
                      chunk_idx: int, frame_cache: dict[int, list[PersonBox]],
                      counted_frames: set[int], tracker,
                      fps: float, stats: DetectionVideoStats) -> tuple[list[dict], bool]:
        """Detect all frames in the chunk. Returns (rows, chunk_ok)."""
        intervals = _union_intervals(chunk_segs)
        rows: list[dict] = []
        chunk_ok = True

        cap = cv2.VideoCapture(video_path)
        batch_size = int(self.cfg.get("yolo.batch_size", 16))
        
        try:
            for (start, end, action_ids) in intervals:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
                curr_fid = start
                
                while curr_fid <= end:
                    batch_fids = []
                    batch_frames = []
                    batch_from_cache = []
                    
                    while len(batch_fids) < batch_size and curr_fid <= end:
                        read_ok, frame = cap.read()
                        if not read_ok:
                            # Unable to read frame → hard fail
                            self._count_frame(curr_fid, False, [], counted_frames, stats)
                            row = _make_row(curr_fid, fps, action_ids, chunk_idx,
                                            [], False, "no_detect")
                            rows.append(row)
                            chunk_ok = False
                            curr_fid += 1
                            continue
                            
                        batch_fids.append(curr_fid)
                        batch_frames.append(frame)
                        batch_from_cache.append(curr_fid in frame_cache)
                        curr_fid += 1
                        
                    if not batch_fids:
                        continue
                        
                    # 1. Run YOLO batch for uncached frames
                    detect_frames = [frame for frame, is_cached in zip(batch_frames, batch_from_cache) if not is_cached]
                    batch_boxes = []
                    if detect_frames:
                        batch_boxes = self._get_yolo().detect_batch(detect_frames, batch_size=batch_size)
                        
                    # 2. Sequential processing (Tracker, Retry, Stats)
                    b_idx = 0
                    for fid, frame, is_cached in zip(batch_fids, batch_frames, batch_from_cache):
                        # Determine the MINIMUM number of people to find
                        if self.dataset == "PKU" and any(aid in self.interactive_ids for aid in action_ids):
                            curr_n_expected = NUM_PERSONS
                        else:
                            curr_n_expected = 1

                        if is_cached:
                            boxes = frame_cache[fid]
                        else:
                            initial_boxes = batch_boxes[b_idx]
                            b_idx += 1
                            # Pass initial_boxes to attempt 0, function will auto-retry if fail
                            boxes = self._detect_with_retry(
                                frame, initial_boxes, tracker, curr_n_expected, stats)
                            frame_cache[fid] = boxes

                        valid = [b for b in boxes if b.confidence >= self.conf_threshold]
                        frame_ok = (len(valid) >= curr_n_expected)

                        if not is_cached:
                            self._count_frame(fid, frame_ok, valid, counted_frames, stats)

                        if not frame_ok:
                            chunk_ok = False

                        bbox_out = [[*b.xyxy, round(b.confidence, 4)] for b in boxes] if boxes else []
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
        """Update stats for frame, only once (skip if counted)."""
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
                           initial_boxes: list[PersonBox] | None,
                           tracker, n_expected: int,
                           stats: DetectionVideoStats) -> list[PersonBox]:
        """YOLO detect with retry. Returns best boxes.

        Attempt 0 (original): default imgsz (uses initial_boxes from batch)
        Retry 1: imgsz=imgsz_retry (1280) + augment=tta
        Retry 2: preprocess image (brightness / CLAHE) + default imgsz
        """
        best_boxes: list[PersonBox] = []
        best_valid_count = 0
        best_max_conf = -1.0

        # Attempt 0 — original
        attempt_boxes = initial_boxes if initial_boxes is not None else self._get_yolo().detect(frame)
        best_boxes, best_valid_count, best_max_conf = self._pick_best(
            attempt_boxes, best_boxes, best_valid_count, best_max_conf)

        if best_valid_count >= n_expected:
            return self._apply_tracker(best_boxes, frame, tracker, n_expected)

        # Retry 1 — larger imgsz + TTA
        if self.max_retries >= 1:
            retry_boxes1 = self._get_yolo().detect(
                frame, imgsz=self.imgsz_retry, augment=self.tta)
            best_boxes, best_valid_count, best_max_conf = self._pick_best(
                retry_boxes1, best_boxes, best_valid_count, best_max_conf)
            stats.retry_count += 1
            if best_valid_count >= n_expected:
                return self._apply_tracker(best_boxes, frame, tracker, n_expected)

        # Retry 2 — preprocess the image
        if self.max_retries >= 2:
            preproc = _apply_retry_preproc(
                frame, self.brightness_delta, self.clahe_on_retry)
            retry_boxes2 = self._get_yolo().detect(preproc)
            best_boxes, best_valid_count, best_max_conf = self._pick_best(
                retry_boxes2, best_boxes, best_valid_count, best_max_conf)
            stats.retry_count += 1

        return self._apply_tracker(best_boxes, frame, tracker, n_expected)

    def _pick_best(self, new_boxes: list[PersonBox],
                   cur_best: list[PersonBox],
                   cur_count: int,
                   cur_conf: float) -> tuple[list[PersonBox], int, float]:
        """Compare and keep better results (more valid boxes, or higher conf)."""
        valid_new = [b for b in new_boxes if b.confidence >= self.conf_threshold]
        max_conf_new = max((b.confidence for b in valid_new), default=0.0)
        # Priority: more valid boxes; if equal -> higher conf
        if (len(valid_new) > cur_count
                or (len(valid_new) == cur_count and max_conf_new > cur_conf)):
            return new_boxes, len(valid_new), max_conf_new
        return cur_best, cur_count, cur_conf

    def _apply_tracker(self, boxes: list[PersonBox],
                       frame: np.ndarray,
                       tracker, n_expected: int) -> list[PersonBox]:
        """Apply tracker (only if n_expected == 2). Returns sorted/sliced ​​boxes."""
        if tracker is not None and boxes:
            return tracker.update(boxes, frame)[:NUM_PERSONS]
        return boxes[:n_expected]
