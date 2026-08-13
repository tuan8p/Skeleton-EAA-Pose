"""Dieu phoi pipeline (PLAN + cac cap nhat):

- Overlap: vung giao giua 2 action extract DUNG 1 LAN, luu buffer dung chung;
  action sau tiep tuc tu frame sau vung giao. Moi action giu nguyen [start, end].
- Moi frame deu qua YOLO26n (detect person) -> crop -> BlazePose. Chi 8 action
  tuong tac PKU bat DeepOCSORT (2 person); con lai top-1 box (single person).
- Temporal Logic: frame trong ngan giua 2 frame conf cao -> FN (retry detect toi
  da 2 lan, van trong -> noi suy bbox theo thoi gian -> BlazePose lai);
  chuoi trong >= `temporal.empty_run_frames` -> TN: skeleton = 0, KHONG loi.
- ok_frames: frame du joints SAU retry + noi suy (TN luon ok; tuong tac: du 2).
- Depth (tuy chon): ghost legs masking truoc BlazePose + back-projection 3D
  goc camera (output.coordinate_mode: "camera").
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .action_segment import ActionSegment
from .annotation_reader import get_annotation_reader
from .config_manager import ConfigManager
from .depth_processor import DepthProcessor
from .image_preprocessor import ImagePreprocessor
from .logger import PipelineLogger, VideoStats
from .mapper import Mapper, NUM_NTU_JOINTS
from .pose_extractor import PoseExtractor, PoseResult
from .progress_manager import ProgressManager
from .skeleton_interpolator import SkeletonInterpolator
from .skeleton_saver import SkeletonSaver
from .skeleton_scaler import SkeletonScaler
from .tracker import get_person_tracker
from .validator import ERR_LOW_CONF, ERR_NO_DETECT, ERR_POSE_FAIL, Validator
from .yolo_detector import PersonBox, YOLODetector

NUM_PERSONS = 2


class _FrameDetect:
    """Ket qua detect 1 frame o pha 1."""
    __slots__ = ("fid", "boxes", "interpolated", "is_tn")

    def __init__(self, fid: int, boxes: list[PersonBox]):
        self.fid = fid
        self.boxes = boxes
        self.interpolated = False   # bbox duoc noi suy thoi gian (FN)
        self.is_tn = False          # thuc su khong co nguoi (TN)


class PipelineOrchestrator:
    def __init__(self, config_path: str | Path | None = None,
                 cfg: ConfigManager | None = None,
                 pose_extractor=None, yolo_detector=None):
        self.cfg = cfg or ConfigManager(config_path)
        self.reader = get_annotation_reader(self.cfg)
        self.preprocessor = ImagePreprocessor(self.cfg)
        self.pose = pose_extractor if pose_extractor is not None else PoseExtractor(self.cfg)
        self.validator = Validator(self.cfg)
        self.interpolator = SkeletonInterpolator(self.cfg)
        self.scaler = SkeletonScaler(self.cfg)
        self.saver = SkeletonSaver(self.cfg)
        self.depth = DepthProcessor(self.cfg)
        self.detection_dir = Path(self.cfg.get("detection.output_dir", "outputs_detect"))
        output_dir = Path(self.cfg.get("paths.output_dir"))
        self.progress = ProgressManager(output_dir / "progress.json")
        self.max_retries = int(self.cfg.get("validator.max_retries", 2))
        self.coord_mode = str(self.cfg.get("output.coordinate_mode", "world"))
        self.save_2d = bool(self.cfg.get("output.save_2d_in_jsonl", True))
        self.interactive_ids = set(self.cfg.get("pku.interactive_action_ids", []))
        self.max_segments = self.cfg.get("runtime.max_segments_per_video", None)
        self.is_pku = str(self.cfg.get("dataset", "PKU")).upper() == "PKU"
        self.empty_run = int(self.cfg.get("temporal.empty_run_frames", 30))
        self.neighbor_conf = self.cfg.get("temporal.neighbor_conf", None)
        if self.neighbor_conf is None:
            self.neighbor_conf = float(self.cfg.get("thresholds.confidence", 0.5))
        self.bbox_interp_max_gap = int(self.cfg.get("temporal.bbox_interp_max_gap", 30))
        self.bbox_buffer_ratio = float(self.cfg.get("pose.bbox_buffer_ratio", 0.1))

    # ---------------- public ----------------
    def run_batch(self, start: int = 0, end: int | None = None) -> None:
        videos = self.reader.list_videos()
        end = len(videos) if end is None else min(end, len(videos))
        logger = PipelineLogger(self.cfg, start, end)
        for video_name in videos[start:end]:
            if self.progress.is_video_done(video_name):
                logger.info(f"Bo qua {video_name} (da xu ly xong)")
                continue
            stats = self.process_video(video_name)
            logger.write_video_log(stats, self.cfg.as_dict())
            logger.append_batch_summary(stats)
            logger.info(f"Xong {video_name}: ok={stats.ok_frames}/{stats.total_frames}, "
                        f"video_ok={stats.video_ok}")
        self.close()

    def process_video(self, video_name: str, disable_pbar: bool = False) -> VideoStats:
        stats = VideoStats(video_name)
        segments = self.reader.read(video_name)
        if self.max_segments:
            segments = segments[: int(self.max_segments)]
        video_path = segments[0].video_path if segments else ""
        has_depth = self.depth.open(video_name)
        if self.coord_mode == "camera" and not has_depth:
            import logging
            logging.getLogger("pipeline").warning(
                f"{video_name}: coordinate_mode=camera nhung khong co depth -> "
                "fallback BlazePose world")
        
        segs = sorted(segments, key=lambda s: s.start_frame)
        intervals = self._union_intervals(segs)
        
        skeleton, meta_rows, fps = self._process_chunk(
            video_name, video_path, intervals, stats, has_depth, disable_pbar)
        
        skeleton, filled = self.interpolator.interpolate(skeleton)
        stats.interpolated_frames.extend(
            row["frame_id"] for row, f in zip(meta_rows, filled) if f)
        
        skeleton = self.scaler.scale(skeleton)
        oks = self._evaluate_chunk(skeleton, intervals, meta_rows)
        for ok, row in zip(oks, meta_rows):
            row["ok"] = ok
            
        stats.ok_frames += sum(oks)
        stats.bad_frames.extend(row["frame_id"] for ok, row in zip(oks, meta_rows) if not ok)
        stats.video_fps = fps
        
        self.saver.save_video(video_name, skeleton, meta_rows)
        
        stats.finish()
        self.depth.close()
        self.progress.mark_video_done(video_name)
        return stats

    def close(self) -> None:
        close = getattr(self.pose, "close", None)
        if callable(close):
            close()
        self.depth.close()

    @staticmethod
    def _union_intervals(segs: list[ActionSegment]) -> list[tuple[int, int, list[int]]]:
        """Hop nhat cac [start,end] chong nhau -> list (start, end, [action_ids]).

        Vung giao mang label list; moi frame chi xuat hien 1 lan (extract 1 lan,
        dung chung cho cac action phu).
        """
        events: list[tuple[int, int, int]] = []  # (pos, +1/-1, action_id)
        for s in segs:
            events.append((s.start_frame, 1, s.action_id))
            events.append((s.end_frame + 1, -1, s.action_id))
        events.sort(key=lambda e: (e[0], -e[1]))
        intervals: list[tuple[int, int, list[int]]] = []
        active: list[int] = []
        prev = None
        for pos, delta, aid in events:
            if prev is not None and pos > prev and active:
                intervals.append((prev, pos - 1, sorted(active)))
            if delta == 1:
                active.append(aid)
            else:
                active.remove(aid)
            prev = pos
        return intervals

    # ---------------- pha 1: detect + temporal FN/TN ----------------
    def _detect_pass(self, video_name: str, video_path: str, intervals, stats: VideoStats,
                     label_is_interactive: list[bool]) -> list[_FrameDetect]:
        import json
        jsonl_path = self.detection_dir / ("PKU" if self.is_pku else "TSU") / f"{video_name}.jsonl"
        frame_data = {}
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        frame_data[d["frame_id"]] = d
                    except:
                        pass
        else:
            import logging
            logging.getLogger("pipeline").warning(f"Khong tim thay file JSONL detect: {jsonl_path}")

        detects: list[_FrameDetect] = []
        for (start, end, _ids), interactive in zip(intervals, label_is_interactive):
            for fid in range(start, end + 1):
                d = frame_data.get(fid)
                boxes = []
                if d and d.get("bboxes"):
                    for b in d["bboxes"]:
                        if len(b) >= 5:
                            boxes.append(PersonBox((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(b[4])))
                stats.detect_total += 1
                if boxes:
                    stats.detect_ok += 1
                    stats.detect_confs.append(max(b.confidence for b in boxes))
                
                valid = [b for b in boxes if b.confidence >= self.neighbor_conf]
                if not interactive and valid:
                    valid = [max(valid, key=lambda b: b.confidence)]
                detects.append(_FrameDetect(fid, valid[:NUM_PERSONS]))
        
        self._classify_empty_runs(detects, stats)
        return detects

    def _classify_empty_runs(self, detects: list[_FrameDetect], stats: VideoStats) -> None:
        """Chuoi trong >= empty_run -> TN; chuoi ngan giua 2 lan can conf cao -> FN
        (noi suy bbox thoi gian)."""
        n = len(detects)
        t = 0
        while t < n:
            if detects[t].boxes:
                t += 1
                continue
            s = t
            while t < n and not detects[t].boxes:
                t += 1
            e = t - 1  # chuoi trong [s..e]
            length = e - s + 1
            if length >= self.empty_run:
                for i in range(s, e + 1):
                    detects[i].is_tn = True
                    stats.tn_frames.append(detects[i].fid)
                continue
            prev_ok = s > 0 and bool(detects[s - 1].boxes)
            next_ok = e + 1 < n and bool(detects[e + 1].boxes)
            if (prev_ok or next_ok) and length <= self.bbox_interp_max_gap:
                for i in range(s, e + 1):
                    stats.fn_frames.append(detects[i].fid)
                self._interpolate_bboxes(detects, s, e)
            # chuoi ngan khong co lan can tot -> de nguyen (loi no_detect luc pose)

    @staticmethod
    def _interpolate_bboxes(detects: list[_FrameDetect], s: int, e: int) -> None:
        """Noi suy tuyen tinh bbox giua frame truoc va sau chuoi trong (theo slot)."""
        prev_boxes = detects[s - 1].boxes if s > 0 else []
        next_boxes = detects[e + 1].boxes if e + 1 < len(detects) else []
        n_slots = max(len(prev_boxes), len(next_boxes))
        for i in range(s, e + 1):
            alpha = (i - (s - 1)) / (e - s + 2)
            boxes: list[PersonBox] = []
            for slot in range(n_slots):
                pb = prev_boxes[slot] if slot < len(prev_boxes) else None
                nb = next_boxes[slot] if slot < len(next_boxes) else None
                ref = pb or nb
                if ref is None:
                    continue
                if pb and nb:
                    xyxy = tuple(int(round(p + (q - p) * alpha))
                                 for p, q in zip(pb.xyxy, nb.xyxy))
                    conf = pb.confidence + (nb.confidence - pb.confidence) * alpha
                else:
                    xyxy, conf = ref.xyxy, ref.confidence
                boxes.append(PersonBox(xyxy=xyxy, confidence=conf,
                                       track_id=ref.track_id))
            detects[i].boxes = boxes
            detects[i].interpolated = True

    # ---------------- pha 2: pose per frame ----------------
    def _process_chunk(self, video_name: str, video_path: str,
                       intervals: list[tuple[int, int, list[int]]],
                       stats: VideoStats, has_depth: bool, disable_pbar: bool = False):
        label_is_interactive = [any(a in self.interactive_ids for a in ids)
                                for _, _, ids in intervals]
        detects = self._detect_pass(video_name, video_path, intervals, stats, label_is_interactive)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = sum(e - s + 1 for s, e, _ in intervals)
        skel = np.full((total, NUM_PERSONS, NUM_NTU_JOINTS, 3), np.nan, dtype=np.float32)
        meta_rows: list[dict] = []
        t = 0
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, desc=f"Pose ({video_name})", position=1, leave=False, disable=disable_pbar)
        
        for (start, end, ids) in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
            interactive = any(a in self.interactive_ids for a in ids)
            for fid in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    while fid <= end:
                        stats.total_frames += 1
                        stats.error_frames[ERR_NO_DETECT].append(fid)
                        meta_rows.append(self._meta_row(
                            fid, fps, ids, {ERR_NO_DETECT: True}, None, None, None))
                        t += 1
                        fid += 1
                    break
                det = detects[t]
                depth_map = None
                if has_depth:
                    raw = self.depth.read(fid, to_meters=(self.coord_mode == "camera"))
                    if raw is not None:
                        depth_map = self.depth.align(raw, (frame.shape[1], frame.shape[0]))
                skel[t], meta = self._process_frame(
                    video_name, frame, det, fid, fps, ids, interactive, stats, depth_map)
                meta_rows.append(meta)
                t += 1
                pbar.update(1)
        pbar.close()
        cap.release()
        return skel, meta_rows, fps

    def _process_frame(self, video_name: str, frame: np.ndarray, det: _FrameDetect,
                       fid: int, fps: float, action_ids: list[int], interactive: bool,
                       stats: VideoStats, depth_map: np.ndarray | None):
        stats.total_frames += 1
        if det.is_tn:
            skel = np.zeros((NUM_PERSONS, NUM_NTU_JOINTS, 3), dtype=np.float32)
            return skel, self._meta_row(fid, fps, action_ids, {"no_person": True},
                                        None, None, None)
        # FN: su dung truc tiep bbox tu JSONL, hoac da duoc bbox interpolation, KHONG retry YOLO
        skel, err, extra, lm2d, confs, retries, vis_list = self._pose_frame(
            frame, det, interactive, stats, depth_map)
        stats.retry_count += retries
        flags: dict = dict(extra)
        if det.interpolated:
            flags["bbox_interpolated"] = True
        if err is not None:
            stats.error_frames[err].append(fid)
            flags[err] = True
            self.saver.save_failed_frame(video_name, fid, err, frame,
                                         draw_boxes=det.boxes if err == ERR_LOW_CONF else None)
        else:
            stats.ok_persons += len(confs)
            if confs:
                stats.conf_values.append(float(np.mean(confs)))
        joint_conf = [v.tolist() if v is not None else None for v in vis_list]
        bbox_rows = [[*b.xyxy, round(b.confidence, 4)] for b in det.boxes] \
            if det.boxes else None
        return skel, self._meta_row(fid, fps, action_ids, flags, lm2d,
                                    bbox_rows, joint_conf)

    def _pose_frame(self, frame: np.ndarray, det: _FrameDetect, interactive: bool,
                    stats: VideoStats, depth_map: np.ndarray | None):
        """Chay BlazePose tren crop tung person (top-1 hoac 2 slot tracked)."""
        n_expected = NUM_PERSONS if interactive else 1
        skel = np.full((NUM_PERSONS, NUM_NTU_JOINTS, 3), np.nan, dtype=np.float32)
        if not interactive:
            skel[1] = 0.0  # single person: person 2 vang -> 0
        retries = 0
        vis_list: list = [None] * NUM_PERSONS
        if not det.boxes:
            return skel, ERR_NO_DETECT, {}, None, [], retries, vis_list
        if all(b.confidence < self.validator.conf_threshold for b in det.boxes):
            return skel, ERR_LOW_CONF, {}, None, [], retries, vis_list
        extra: dict = {}
        if interactive and len(det.boxes) < NUM_PERSONS:
            extra["person_missing"] = len(det.boxes)
        lm2d_all = np.full((NUM_PERSONS, NUM_NTU_JOINTS, 2), np.nan, dtype=np.float32)
        confs: list[float] = []
        person_errors: list[str] = []
        for p, box in enumerate(det.boxes[:n_expected]):
            crop, (x0, y0) = box.crop(frame, margin=self.bbox_buffer_ratio)
            if crop.size == 0:
                person_errors.append(ERR_NO_DETECT)
                continue
            if depth_map is not None:
                masked = self.depth.mask_obstruction(frame, [box.xyxy], depth_map)
                if masked is not frame:
                    crop, _ = box.crop(masked)
            res, tfm, r = self._extract_with_retry(crop)
            retries += r
            if res is None:
                extra[f"person{p + 1}_{ERR_NO_DETECT}"] = True
                person_errors.append(ERR_NO_DETECT)
                continue
            skel, err, flags_p, lm2d_p, confs_p, vis25 = self._validate_person(
                res, tfm, frame, (x0, y0), (crop.shape[1], crop.shape[0]), skel, p,
                depth_map)
            vis_list[p] = vis25
            if err is None:
                confs.extend(confs_p)
                for k, v in flags_p.items():
                    extra[f"person{p + 1}_{k}"] = v
            else:
                extra[f"person{p + 1}_{err}"] = True
                person_errors.append(err)
            if lm2d_p is not None:
                lm2d_all[p] = lm2d_p
        if not confs:
            err = ERR_NO_DETECT if all(e == ERR_NO_DETECT for e in person_errors) \
                else ERR_POSE_FAIL
            return skel, err, extra, lm2d_all, [], retries, vis_list
        return skel, None, extra, lm2d_all, confs, retries, vis_list

    def _validate_person(self, res: PoseResult, tfm, frame: np.ndarray,
                         crop_offset: tuple[int, int], crop_wh: tuple[int, int] | None,
                         skel: np.ndarray, person_idx: int,
                         depth_map: np.ndarray | None):
        """Map 33->25, validate 1 person. Tra ve (skel, err|None, flags, lm2d, confs, vis25)."""
        vis25 = Mapper.map_visibility(res.visibility)
        lm2d = self._map_2d(res, tfm, frame.shape[:2], crop_offset, crop_wh)
        skel25 = self._to_output_coords(res, lm2d, frame.shape[:2], depth_map)
        if self.validator.all_low_conf(vis25):
            return skel, ERR_LOW_CONF, {}, lm2d if self.save_2d else None, [], vis25
        val = self.validator.finalize(skel25, vis25)
        if val.error_type == ERR_POSE_FAIL:
            return skel, ERR_POSE_FAIL, {}, lm2d if self.save_2d else None, [], vis25
        skel[person_idx] = val.skeleton
        good = vis25[vis25 >= self.validator.conf_threshold]
        conf = float(np.mean(good)) if len(good) else 0.0
        flags = {"nan_joints": val.nan_joints} if val.nan_joints else {}
        return skel, None, flags, lm2d if self.save_2d else None, [conf], vis25

    def _to_output_coords(self, res: PoseResult, lm2d: np.ndarray,
                          frame_hw: tuple[int, int],
                          depth_map: np.ndarray | None) -> np.ndarray:
        """Chon he toa do output: camera (depth back-projection) | world | image."""
        if self.coord_mode == "camera" and depth_map is not None:
            fh, fw = frame_hw
            lm2d_px = lm2d * np.array([fw, fh], dtype=np.float32)
            return self.depth.backproject(lm2d_px, depth_map)
        src = res.world_landmarks if self.coord_mode in ("world", "camera") \
            else res.image_landmarks
        return Mapper.map_coords(src)

    def _evaluate_chunk(self, skeleton: np.ndarray, intervals, meta_rows) -> list[bool]:
        """Frame ok = khong con NaN (sau noi suy) tren person ky vong; TN luon ok."""
        oks: list[bool] = []
        t = 0
        for start, end, ids in intervals:
            expected = NUM_PERSONS if any(a in self.interactive_ids for a in ids) else 1
            for _ in range(start, end + 1):
                row = meta_rows[t]
                if row["error_flags"].get("no_person"):
                    oks.append(True)  # TN: khong loi
                else:
                    oks.append(bool(not np.isnan(skeleton[t, :expected]).any()))
                t += 1
        return oks

    # ---------------- helpers ----------------
    def _attempts(self, frame: np.ndarray):
        """Cac bien the anh de thu: [raw, retry1, retry2] theo thu tu."""
        h0, w0 = frame.shape[:2]
        yield frame, (1.0, 0, 0, w0, h0)
        if self.max_retries >= 1:
            yield self.preprocessor.process_retry1(frame)
        if self.max_retries >= 2:
            yield self.preprocessor.process_retry2(frame)

    def _extract_with_retry(self, frame: np.ndarray):
        """Tra ve (PoseResult|None, transform, so_lan_retry)."""
        h0, w0 = frame.shape[:2]
        result, best_tfm, retries = None, (1.0, 0, 0, w0, h0), 0
        for img, tfm in self._attempts(frame):
            res = self.pose.process(img)
            if res is None:
                continue
            result, best_tfm = res, tfm
            if not self.validator.needs_retry(res.visibility):
                break
            retries += 1
        return result, best_tfm, retries

    def _map_2d(self, res: PoseResult, tfm, frame_hw: tuple[int, int],
                crop_offset: tuple[int, int],
                crop_wh: tuple[int, int] | None) -> np.ndarray:
        """Landmarks 2D normalized theo frame goc (bun letterbox + crop offset)."""
        fh, fw = frame_hw
        lm2d = Mapper.map_coords(res.image_landmarks)[:, :2]
        if crop_wh is not None:
            cw, ch = crop_wh
            lm2d = ImagePreprocessor.to_original_coords(lm2d, (cw, ch), tfm)
            lm2d[:, 0] = (lm2d[:, 0] * cw + crop_offset[0]) / fw
            lm2d[:, 1] = (lm2d[:, 1] * ch + crop_offset[1]) / fh
        else:
            lm2d = ImagePreprocessor.to_original_coords(lm2d, (fw, fh), tfm)
        return lm2d

    def _meta_row(self, fid: int, fps: float, action_ids,
                  flags: dict, lm2d, bboxes, joint_conf) -> dict:
        if isinstance(action_ids, list):
            label = action_ids[0] if len(action_ids) == 1 else [int(a) for a in action_ids]
        else:
            label = int(action_ids)
        row = {
            "frame_id": int(fid),
            "timestamp": round(fid / fps, 4),
            "action_label": label,
            "error_flags": flags,
        }
        if lm2d is not None and self.save_2d:
            row["landmarks_2d"] = np.round(np.asarray(lm2d), 5).tolist()
        if bboxes is not None:
            row["bboxes"] = bboxes
        if joint_conf is not None and any(j is not None for j in joint_conf):
            row["joint_conf"] = [
                None if j is None else [round(float(x), 4) for x in j]
                for j in joint_conf
            ]
        return row
