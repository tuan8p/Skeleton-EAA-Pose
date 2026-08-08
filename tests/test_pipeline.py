"""Unit tests cho pipeline Skeleton-EAA-Pose. Chay: pytest tests/ -v"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.action_segment import ActionSegment
from src.annotation_reader import PKUAnnotationReader, TSUAnnotationReader
from src.config_manager import ConfigManager
from src.image_preprocessor import ImagePreprocessor
from src.mapper import BLAZE_TO_NTU, Mapper, NUM_NTU_JOINTS
from src.pose_extractor import PoseResult
from src.progress_manager import ProgressManager
from src.skeleton_interpolator import SkeletonInterpolator
from src.skeleton_saver import SkeletonSaver
from src.validator import ERR_LOW_CONF, ERR_NO_DETECT, ERR_POSE_FAIL, Validator
from src.yolo_detector import PersonBox, YOLODetector


def make_cfg(tmp_path: Path, **extra) -> ConfigManager:
    base = {
        "dataset": "PKU",
        "paths.video_dir": str(tmp_path / "videos"),
        "paths.annotation_dir": str(tmp_path / "labels"),
        "paths.output_dir": str(tmp_path / "out"),
        "paths.log_dir": str(tmp_path / "logs"),
        "paths.failed_frames_dir": str(tmp_path / "failed"),
        "preprocessing.resize.enabled": False,
    }
    base.update(extra)
    cfg = ConfigManager(None, overrides=base)
    (tmp_path / "videos").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels").mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------- Config / Segment / Readers ----------------

class TestConfigManager:
    def test_get_dot_path_and_default(self, tmp_path):
        cfg = ConfigManager(None, overrides={"a.b": 1})
        assert cfg.get("a.b") == 1
        assert cfg.get("a.x", 99) == 99

    def test_save_load_roundtrip(self, tmp_path):
        cfg = ConfigManager(None, overrides={"mediapipe.model_complexity": 2})
        p = tmp_path / "c.yaml"
        cfg.save(p)
        cfg2 = ConfigManager(p)
        assert cfg2.get("mediapipe.model_complexity") == 2


class TestActionSegment:
    def test_no_overlap(self):
        seg = ActionSegment("v", "v", 1, 10, 20)
        assert seg.resolve_overlap(-1) == (10, 20)

    def test_overlap_cut(self):
        seg = ActionSegment("v", "v", 1, 10, 20)
        assert seg.resolve_overlap(14) == (15, 20)

    def test_fully_covered(self):
        seg = ActionSegment("v", "v", 1, 10, 20)
        assert seg.resolve_overlap(20) is None


class TestAnnotationReaders:
    def test_pku_read(self, tmp_path):
        (tmp_path / "videos" ).mkdir(); (tmp_path / "labels").mkdir()
        (tmp_path / "videos" / "0001-L.avi").touch()
        (tmp_path / "labels" / "0001-L.txt").write_text(
            "23,1,29,1\n29,38,67,2\n", encoding="utf-8")
        cfg = make_cfg(tmp_path)
        reader = PKUAnnotationReader(cfg)
        assert reader.list_videos() == ["0001-L"]
        segs = reader.read("0001-L")
        assert len(segs) == 2
        assert (segs[0].action_id, segs[0].start_frame, segs[0].end_frame) == (23, 1, 29)
        assert segs[1].confidence == 2.0

    def test_tsu_read_with_event_map(self, tmp_path):
        (tmp_path / "videos").mkdir(); (tmp_path / "labels" / "p01").mkdir(parents=True)
        (tmp_path / "videos" / "scene01.mp4").touch()
        (tmp_path / "labels" / "p01" / "scene01.csv").write_text(
            "event,start_frame,end_frame\nwalk,10,50\nsit,60,90\n", encoding="utf-8")
        cfg = make_cfg(tmp_path, **{"dataset": "TSU",
                                    "tsu.event_map": {"walk": 1, "sit": 2}})
        reader = TSUAnnotationReader(cfg)
        assert reader.list_videos() == ["scene01"]
        segs = reader.read("scene01")
        assert [s.action_id for s in segs] == [1, 2]
        assert (segs[0].start_frame, segs[0].end_frame) == (10, 50)

    def test_tsu_auto_event_map(self, tmp_path):
        """Event map tu dong: moi ten event (ke ca sub-action) la 1 label rieng biet."""
        (tmp_path / "videos").mkdir(); (tmp_path / "labels").mkdir()
        (tmp_path / "videos" / "s.mp4").touch()
        (tmp_path / "labels" / "s.csv").write_text(
            "event,start_frame,end_frame\nWalk,1,9\nMake_coffee,10,50\n"
            "Make_coffee.Get_water,10,30\n", encoding="utf-8")
        cfg = make_cfg(tmp_path, **{"dataset": "TSU"})
        reader = TSUAnnotationReader(cfg)
        segs = reader.read("s")
        assert len(segs) == 3
        # sorted unique: Make_coffee=1, Make_coffee.Get_water=2, Walk=3
        assert [s.action_id for s in segs] == [3, 1, 2]
        map_file = Path(cfg.get("paths.output_dir")) / "tsu_event_map.json"
        assert map_file.exists()
        saved = json.loads(map_file.read_text())
        assert saved == {"Make_coffee": 1, "Make_coffee.Get_water": 2, "Walk": 3}
        # lan chay sau doc lai tu file -> nhat quan
        reader2 = TSUAnnotationReader(make_cfg(tmp_path, **{"dataset": "TSU"}))
        assert [s.action_id for s in reader2.read("s")] == [3, 1, 2]

    def test_tsu_overlap_label_list(self, tmp_path):
        """Sub-action chong len action cha -> action_label dang list."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        (tmp_path / "videos").mkdir(); (tmp_path / "labels").mkdir()
        cfg = make_cfg(tmp_path, **{"dataset": "TSU"})
        _write_test_video(tmp_path / "videos" / "s.mp4", n_frames=60)
        (tmp_path / "labels" / "s.csv").write_text(
            "event,start_frame,end_frame\nWalk,1,30\nWalk.Slow,10,20\n", encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO())
        orch.process_video("s")
        rows = [json.loads(l) for l in
                Path(cfg.get("paths.output_dir"), "chunks", "jsonl",
                     "s_chunk_000.jsonl").read_text().splitlines()]
        # sorted unique: Walk=1, Walk.Slow=2
        assert rows[0]["action_label"] == 1            # frame 1: chi Walk
        mid = rows[10]                                 # frame 11: ca 2 event phu
        assert mid["action_label"] == [1, 2]
        assert rows[25]["action_label"] == 1           # het vung overlap
        assert all(r["ok"] is True for r in rows)


# ---------------- Mapper / Validator ----------------

class TestMapper:
    def test_shape_and_mean(self):
        lm = np.arange(33 * 3, dtype=np.float32).reshape(33, 3)
        out = Mapper.map_coords(lm)
        assert out.shape == (25, 3)
        np.testing.assert_allclose(out[0], lm[[23, 24]].mean(axis=0))  # SpineBase
        np.testing.assert_allclose(out[3], lm[0])                      # Head = nose
        np.testing.assert_allclose(out[7], lm[[17, 19, 21]].mean(axis=0))

    def test_visibility(self):
        vis = np.ones(33, dtype=np.float32)
        vis[23] = 0.0
        out = Mapper.map_visibility(vis)
        assert out.shape == (25,)
        assert out[0] == pytest.approx(0.5)   # mean(hip L=0, hip R=1)
        assert out[12] == 0.0                 # HipLeft = blaze 23
        assert len(BLAZE_TO_NTU) == NUM_NTU_JOINTS


class TestValidator:
    def test_needs_retry(self, tmp_path):
        v = Validator(make_cfg(tmp_path))
        assert v.needs_retry(np.array([0.9, 0.4]))
        assert not v.needs_retry(np.array([0.9, 0.6]))

    def test_finalize_marks_nan_and_pose_fail(self, tmp_path):
        v = Validator(make_cfg(tmp_path, **{"thresholds.max_nan_ratio": 0.4}))
        skel = np.ones((25, 3), dtype=np.float32)
        vis = np.ones(25, dtype=np.float32)
        vis[:11] = 0.1  # 11/25 = 0.44 > 0.4
        out = v.finalize(skel, vis)
        assert out.error_type == ERR_POSE_FAIL
        assert np.isnan(out.skeleton[:11]).all()
        vis2 = np.ones(25, dtype=np.float32); vis2[0] = 0.1
        out2 = v.finalize(skel, vis2)
        assert out2.error_type is None and np.isnan(out2.skeleton[0]).all()


# ---------------- Interpolator ----------------

class TestInterpolator:
    def test_temporal_short_gap_interpolated(self, tmp_path):
        interp = SkeletonInterpolator(make_cfg(tmp_path))
        skel = np.zeros((6, 2, 25, 3), dtype=np.float32)
        skel[:, :, :, 0] = np.arange(6)[None, :, None].T.reshape(6, 1, 1)
        skel[2:4] = np.nan  # gap = 2 <= 3
        out, filled = interp.interpolate(skel)
        assert not np.isnan(out).any()
        assert out[2, 0, 0, 0] == pytest.approx(2.0)
        assert list(filled) == [False, False, True, True, False, False]

    def test_temporal_long_gap_kept_nan(self, tmp_path):
        interp = SkeletonInterpolator(make_cfg(tmp_path))
        skel = np.zeros((10, 2, 25, 3), dtype=np.float32)
        skel[2:7] = np.nan  # gap = 5 > 3
        out, filled = interp.interpolate(skel)
        assert np.isnan(out[2:7]).all()
        assert not filled.any()

    def test_spatial_single_joint(self, tmp_path):
        interp = SkeletonInterpolator(make_cfg(tmp_path))
        skel = np.ones((5, 2, 25, 3), dtype=np.float32)
        skel[:, :, 5, :] = 3.0            # ElbowL = ShoulderL(=1) + offset 2
        skel[2, 0, 5, :] = np.nan         # 1 khop don le bi thieu
        out, filled = interp.interpolate(skel)
        np.testing.assert_allclose(out[2, 0, 5], [3.0, 3.0, 3.0])
        assert filled[2] and not filled[0]


# ---------------- IO: Saver / Progress / Preprocessor / IoU ----------------

class TestSaverProgress:
    def test_save_chunk_files(self, tmp_path):
        cfg = make_cfg(tmp_path)
        saver = SkeletonSaver(cfg)
        skel = np.zeros((4, 2, 25, 3), dtype=np.float32)
        rows = [{"frame_id": 1, "timestamp": 0.03, "action_label": 5, "error_flags": {}}]
        saver.save_chunk("vid", 0, skel, rows)
        assert saver.npy_path("vid", 0).name == "vid_chunk_000.npy"
        assert saver.chunk_exists("vid", 0)
        loaded = np.load(saver.npy_path("vid", 0))
        assert loaded.shape == (4, 2, 25, 3) and loaded.dtype == np.float32
        line = json.loads(saver.jsonl_path("vid", 0).read_text().strip())
        assert line["action_label"] == 5

    def test_save_failed_frame_with_box(self, tmp_path):
        import cv2
        cfg = make_cfg(tmp_path)
        saver = SkeletonSaver(cfg)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        box = PersonBox(xyxy=(2, 2, 10, 10), confidence=0.3)
        saver.save_failed_frame("vid", 7, ERR_LOW_CONF, frame, draw_boxes=[box])
        assert (tmp_path / "failed" / "vid" / "vid_frame_7_low_conf.jpg").exists()
        assert (tmp_path / "failed" / "vid" / "vid_frame_7_d.jpg").exists()

    def test_progress_resume(self, tmp_path):
        p = ProgressManager(tmp_path / "out" / "progress.json")
        p.mark_chunk_done("vid", 0, 2)
        p2 = ProgressManager(tmp_path / "out" / "progress.json")
        assert p2.is_chunk_done("vid", 0) and not p2.is_chunk_done("vid", 1)
        p2.mark_video_done("vid")
        assert ProgressManager(tmp_path / "out" / "progress.json").is_video_done("vid")


class TestPreprocessorAndIoU:
    def test_letterbox_roundtrip(self, tmp_path):
        cfg = make_cfg(tmp_path, **{"preprocessing.resize.enabled": True,
                                    "preprocessing.resize.width": 640,
                                    "preprocessing.resize.height": 640})
        pre = ImagePreprocessor(cfg)
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        out, tfm = pre.process(frame)
        assert out.shape == (640, 640, 3)
        xy = np.array([[0.5, 0.5]], dtype=np.float32)
        orig = ImagePreprocessor.to_original_coords(xy, (854, 480), tfm, (640, 640))
        assert orig[0, 0] == pytest.approx(0.5, abs=1e-3)

    def test_iou_tracking_order(self):
        prev = [PersonBox((0, 0, 10, 10), 0.9), PersonBox((100, 100, 120, 120), 0.8)]
        curr = [PersonBox((101, 101, 121, 121), 0.95), PersonBox((1, 1, 11, 11), 0.7)]
        ordered = YOLODetector.match_by_iou(prev, curr)
        assert ordered[0].xyxy == (1, 1, 11, 11)
        assert ordered[1].xyxy == (101, 101, 121, 121)

    def test_crop_offset(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        box = PersonBox((50, 30, 100, 80), 0.9)
        crop, (x0, y0) = box.crop(frame, margin=0.0)
        assert crop.shape[:2] == (50, 50) and (x0, y0) == (50, 30)


# ---------------- End-to-end voi mock ----------------

class FakePose:
    """Gia lap BlazePose: landmarks khac nhau theo mau frame (de test overlap dung
    chung); fail khi mau frame thuoc blacklist."""
    def __init__(self, fail_colors: set[int] | None = None):
        self.fail_colors = fail_colors or set()
        self.calls = 0

    def process(self, img):
        self.calls += 1
        color = float(img.mean())
        if any(abs(color - c) <= 0.6 for c in self.fail_colors):
            return None
        rng = np.random.default_rng(int(round(color)) % 1000)
        world = rng.normal(0, 0.5, (33, 3)).astype(np.float32)
        image = rng.uniform(0, 1, (33, 3)).astype(np.float32)
        vis = np.full(33, 0.99, dtype=np.float32)
        return PoseResult(world_landmarks=world, image_landmarks=image, visibility=vis)

    def close(self):
        pass


class FakeYOLO:
    """Gia lap YOLO: 2 box nguoi; rong khi mau frame thuoc empty_colors."""
    available = True

    def __init__(self, empty_colors: set[int] | None = None):
        self.empty_colors = empty_colors or set()

    def detect(self, frame):
        color = float(frame.mean())
        if any(abs(color - c) <= 0.6 for c in self.empty_colors):
            return []
        h, w = frame.shape[:2]
        return [PersonBox((0, 0, w // 2, h), 0.9), PersonBox((w // 2, 0, w - 1, h), 0.8)]


def _write_test_video(path: Path, n_frames: int = 60, w: int = 64, h: int = 48):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG") if path.suffix == ".avi" \
        else cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    writer.release()


class TestOrchestratorE2E:
    @staticmethod
    def _setup_video(tmp_path):
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path, **{"pku.interactive_action_ids": [12]})
        _write_test_video(tmp_path / "videos" / "vtest.avi", n_frames=60)
        (tmp_path / "labels" / "vtest.txt").write_text(
            "1,1,20,2\n5,30,45,2\n12,50,60,2\n", encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO())
        return orch, cfg

    def test_end_to_end(self, tmp_path):
        orch, cfg = self._setup_video(tmp_path)
        stats = orch.process_video("vtest")
        # 47 frame (20+16+11); ok_frames dem theo FRAME; ok_persons theo person
        assert stats.total_frames == 47
        assert stats.ok_frames == 47
        assert stats.ok_persons == 36 + 11 * 2
        npy = np.load(Path(cfg.get("paths.output_dir"),
                           "chunks", "npy", "vtest_chunk_000.npy"))
        assert npy.shape == (47, 2, 25, 3)
        assert not np.isnan(npy).any()
        rows = [json.loads(l) for l in
                Path(cfg.get("paths.output_dir"), "chunks", "jsonl",
                     "vtest_chunk_000.jsonl").read_text().splitlines()]
        assert len(rows) == 47
        assert rows[0]["action_label"] == 1 and rows[0]["frame_id"] == 1
        assert rows[36]["action_label"] == 12 and rows[36]["frame_id"] == 50
        assert "landmarks_2d" in rows[0]
        # metadata moi: ok flag + bboxes (frame tuong tac) + joint_conf
        assert all(r["ok"] is True for r in rows)
        assert "bboxes" in rows[36] and len(rows[36]["bboxes"]) == 2
        assert rows[0]["joint_conf"][0] is not None and rows[0]["joint_conf"][1] is None
        assert stats.video_ok is True
        # segment 1 nguoi: person 2 = 0; segment tuong tac: person 2 co du lieu
        assert (npy[0, 1] == 0).all()
        assert not np.isnan(npy[40, 1]).all() and not (npy[40, 1] == 0).all()
        assert ProgressManager(Path(cfg.get("paths.output_dir")) / "progress.json") \
            .is_video_done("vtest")

    def test_chunking_and_resume(self, tmp_path):
        from src.pipeline_orchestrator import PipelineOrchestrator
        orch, cfg = self._setup_video(tmp_path)
        orch.cfg.set("chunk.max_actions_per_chunk", 2)
        orch.process_video("vtest")
        out = Path(cfg.get("paths.output_dir")) / "chunks" / "npy"
        assert (out / "vtest_chunk_000.npy").exists() and (out / "vtest_chunk_001.npy").exists()
        a0 = np.load(out / "vtest_chunk_000.npy")
        a1 = np.load(out / "vtest_chunk_001.npy")
        assert a0.shape[0] == 36 and a1.shape[0] == 11  # seg(1..20)+(30..45), seg(50..60)
        # chay lai: chunk da done -> khong ghi de (mtime giu nguyen)
        t0 = (out / "vtest_chunk_000.npy").stat().st_mtime_ns
        orch2 = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                     yolo_detector=FakeYOLO())
        orch2.process_video("vtest")
        assert (out / "vtest_chunk_000.npy").stat().st_mtime_ns == t0

    def test_no_detect_marks_error(self, tmp_path):
        """3 frame dau pose None (co bbox nhung BlazePose fail) -> no_detect x3;
        noi suy lap duoc (gap=3) -> ok_frames=10, video_ok=True."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vfail.avi", n_frames=10)
        (tmp_path / "labels" / "vfail.txt").write_text("3,1,10,1\n", encoding="utf-8")
        # frame 1,2,3 co mau 0,1,2 -> FakePose None
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(fail_colors={0, 1, 2}),
                                    yolo_detector=FakeYOLO())
        stats = orch.process_video("vfail")
        assert len(stats.error_frames[ERR_NO_DETECT]) == 3   # loi luc extract
        assert stats.ok_persons == 7                          # person hop le luc extract
        assert stats.ok_frames == 10                          # sau noi suy: du joints
        assert stats.bad_frames == [] and stats.video_ok is True
        imgs = list((Path(cfg.get("paths.failed_frames_dir")) / "vfail").glob("*_no_detect.jpg"))
        assert len(imgs) == 3

    def test_bad_frames_when_gap_too_long(self, tmp_path):
        """4 frame loi lien tiep (> max_gap=3) -> khong noi suy duoc -> bad frame,
        video_ok=False, ok_frames giam tuong ung."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vfail2.avi", n_frames=10)
        (tmp_path / "labels" / "vfail2.txt").write_text("3,1,10,1\n", encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg,
                                    pose_extractor=FakePose(fail_colors={0, 1, 2, 3}),
                                    yolo_detector=FakeYOLO())
        stats = orch.process_video("vfail2")
        assert len(stats.error_frames[ERR_NO_DETECT]) == 4
        assert stats.ok_frames == 6
        assert stats.bad_frames == [1, 2, 3, 4]
        assert stats.video_ok is False
        rows = [json.loads(l) for l in
                Path(cfg.get("paths.output_dir"), "chunks", "jsonl",
                     "vfail2_chunk_000.jsonl").read_text().splitlines()]
        assert [r["ok"] for r in rows[:4]] == [False] * 4
        assert all(r["ok"] is True for r in rows[4:])

    def test_fn_bbox_interpolation(self, tmp_path):
        """FN: frame 5 trong giua 2 frame lan can conf cao -> retry detect van trong
        -> noi suy bbox -> pose thanh cong -> KHONG loi, flag bbox_interpolated."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vfn.avi", n_frames=10)
        (tmp_path / "labels" / "vfn.txt").write_text("3,1,10,1\n", encoding="utf-8")
        # frame 5 co mau 4 -> YOLO trong
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO(empty_colors={4}))
        stats = orch.process_video("vfn")
        assert stats.fn_frames == [5]
        assert stats.tn_frames == []
        assert sum(len(v) for v in stats.error_frames.values()) == 0
        assert stats.ok_frames == 10 and stats.video_ok is True
        rows = [json.loads(l) for l in
                Path(cfg.get("paths.output_dir"), "chunks", "jsonl",
                     "vfn_chunk_000.jsonl").read_text().splitlines()]
        assert rows[4]["error_flags"].get("bbox_interpolated") is True
        assert rows[4]["ok"] is True

    def test_tn_empty_room(self, tmp_path):
        """TN: chuoi trong 30 frame (>= empty_run_frames) -> skeleton=0, khong loi,
        khong noi suy, ok=True."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vtn.avi", n_frames=40)
        (tmp_path / "labels" / "vtn.txt").write_text("3,1,40,1\n", encoding="utf-8")
        # frame 5..34 (mau 4..33) trong -> 30 frame lien tiep
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO(empty_colors=set(range(4, 34))))
        stats = orch.process_video("vtn")
        assert len(stats.tn_frames) == 30 and stats.fn_frames == []
        assert sum(len(v) for v in stats.error_frames.values()) == 0
        assert stats.ok_frames == 40 and stats.video_ok is True
        npy = np.load(Path(cfg.get("paths.output_dir"), "chunks", "npy",
                           "vtn_chunk_000.npy"))
        assert (npy[4:34] == 0).all()          # vung TN skeleton = 0
        assert not np.isnan(npy).any()
        rows = [json.loads(l) for l in
                Path(cfg.get("paths.output_dir"), "chunks", "jsonl",
                     "vtn_chunk_000.jsonl").read_text().splitlines()]
        assert rows[4]["error_flags"].get("no_person") is True
        assert rows[4]["ok"] is True

    def test_overlap_shared_between_actions(self, tmp_path):
        """Vung giao 2 action extract 1 lan, dung chung: ca 2 file action deu chua
        day du dai frame cua minh, vung overlap giong het nhau."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vov.avi", n_frames=35)
        (tmp_path / "labels" / "vov.txt").write_text("1,1,20,1\n5,15,30,1\n",
                                                     encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO())
        orch.process_video("vov")
        out = Path(cfg.get("paths.output_dir"))
        npy = np.load(out / "chunks" / "npy" / "vov_chunk_000.npy")
        assert npy.shape[0] == 30                    # union 1..30, khong trung lap
        rows = [json.loads(l) for l in
                (out / "chunks" / "jsonl" / "vov_chunk_000.jsonl")
                .read_text().splitlines()]
        assert rows[0]["action_label"] == 1
        assert rows[14]["action_label"] == [1, 5]    # frame 15: overlap
        assert rows[25]["action_label"] == 5
        from scripts.split_actions import split_video
        results = split_video(out, "vov")
        assert [r[2] for r in results] == [20, 16]   # action du dai [start,end]
        a0 = np.load(out / "actions" / "npy" / "vov_000.npy")
        a1 = np.load(out / "actions" / "npy" / "vov_001.npy")
        np.testing.assert_allclose(a0[14:20], a1[0:6])   # vung overlap dung chung

    def test_run_batch_writes_logs(self, tmp_path):
        orch, cfg = self._setup_video(tmp_path)
        orch.run_batch(0, 1)
        log_dir = Path(cfg.get("paths.log_dir"))
        assert (log_dir / "vtest.log").exists()
        assert list(log_dir.glob("batch_summary_0_1.csv"))


class TestScaler:
    def test_scale_invariant_to_camera_distance(self, tmp_path):
        """Nhan toan bo toa do voi hang so (doi tuong xa/gan) -> ket qua giong nhau."""
        from src.skeleton_scaler import SkeletonScaler
        scaler = SkeletonScaler(make_cfg(tmp_path))
        rng = np.random.default_rng(1)
        base = rng.normal(size=(5, 2, 25, 3)).astype(np.float32)
        a = scaler.scale(base)
        b = scaler.scale(base * 7.5)
        np.testing.assert_allclose(a, b, rtol=1e-5)

    def test_spine_mean_becomes_one(self, tmp_path):
        from src.skeleton_scaler import SkeletonScaler
        scaler = SkeletonScaler(make_cfg(tmp_path))
        skel = np.zeros((3, 2, 25, 3), dtype=np.float32)
        skel[:, 0, 1, 1] = 2.0  # SpineMid cach SpineBase 2 don vi
        out = scaler.scale(skel)
        assert out[0, 0, 1, 1] == pytest.approx(1.0)

    def test_nan_and_zero_person_untouched(self, tmp_path):
        from src.skeleton_scaler import SkeletonScaler
        scaler = SkeletonScaler(make_cfg(tmp_path))
        skel = np.zeros((4, 2, 25, 3), dtype=np.float32)
        skel[:, 0, 1, 1] = 2.0
        skel[1, 0] = np.nan
        out = scaler.scale(skel)
        assert np.isnan(out[1, 0]).all()     # NaN giu nguyen
        assert (out[:, 1] == 0).all()        # person vang giu 0

    def test_disabled(self, tmp_path):
        from src.skeleton_scaler import SkeletonScaler
        scaler = SkeletonScaler(make_cfg(tmp_path, **{"scaling.enabled": False}))
        skel = np.ones((2, 2, 25, 3), dtype=np.float32)
        assert scaler.scale(skel) is skel


class TestSplitActions:
    def test_split_per_action(self, tmp_path):
        from scripts.split_actions import split_video
        orch, cfg = TestOrchestratorE2E._setup_video(tmp_path)
        orch.process_video("vtest")
        out = Path(cfg.get("paths.output_dir"))
        results = split_video(out, "vtest")
        assert [r[1] for r in results] == [1, 5, 12]      # action ids theo thu tu
        assert [r[2] for r in results] == [20, 16, 11]    # so frame moi action
        a0 = np.load(out / "actions" / "npy" / "vtest_000.npy")
        assert a0.shape == (20, 2, 25, 3)
        rows = (out / "actions" / "jsonl" / "vtest_002.jsonl").read_text().strip().splitlines()
        assert len(rows) == 11


class TestDemoOverlay:
    def test_overlay_video_created(self, tmp_path):
        from src.demo_overlay import create_overlay_video
        video = tmp_path / "videos" / "vtest.avi"
        _write_test_video(video, n_frames=10)
        skel = np.zeros((10, 2, 25, 3), dtype=np.float32)
        np.save(tmp_path / "s.npy", skel)
        lm = np.zeros((2, 25, 2), dtype=np.float32) + 0.5
        with open(tmp_path / "s.jsonl", "w") as f:
            for i in range(1, 11):
                f.write(json.dumps({"frame_id": i, "timestamp": i / 30,
                                    "action_label": 1, "error_flags": {},
                                    "landmarks_2d": lm.tolist()}) + "\n")
        out = tmp_path / "overlay.mp4"
        create_overlay_video(str(video), str(tmp_path / "s.npy"),
                             str(tmp_path / "s.jsonl"), str(out))
        assert out.exists() and out.stat().st_size > 0


class TestDepthProcessor:
    def _make_processor(self, tmp_path, **extra):
        from src.depth_processor import DepthProcessor
        cfg = make_cfg(tmp_path, **{"depth.enabled": True, **extra})
        return DepthProcessor(cfg)

    def test_backproject_formula(self, tmp_path):
        """X = (x-cx)*Z/fx ... voi Z lay tu depth map."""
        proc = self._make_processor(tmp_path)
        proc.intr = {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0}
        depth = np.full((480, 640), 2000.0, dtype=np.float32)  # 2000 (don vi map)
        lm2d_px = np.array([[320.0, 240.0], [420.0, 340.0]], dtype=np.float32)
        xyz = proc.backproject(lm2d_px, depth)
        np.testing.assert_allclose(xyz[0], [0.0, 0.0, 2000.0])
        np.testing.assert_allclose(xyz[1], [2000.0, 2000.0, 2000.0])

    def test_backproject_bad_z_uses_parent(self, tmp_path):
        """Z=0 tai khop con -> bu bang Z khop cha (ElbowL=5 cha la ShoulderL=4)."""
        proc = self._make_processor(tmp_path)
        proc.intr = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}
        depth = np.full((100, 100), 1000.0, dtype=np.float32)
        depth[10, 10] = 0.0  # mat depth tai joint 5
        lm2d_px = np.zeros((25, 2), dtype=np.float32)
        lm2d_px[4] = [50.0, 50.0]
        lm2d_px[5] = [10.0, 10.0]
        xyz = proc.backproject(lm2d_px, depth)
        assert xyz[5, 2] == pytest.approx(1000.0)   # bu tu cha
        assert not np.isnan(xyz[5]).any()

    def test_ghost_legs_masking(self, tmp_path):
        """Vat can (depth gan hon nguoi) chiem >1% bbox -> to xam; khong co -> giu nguyen."""
        proc = self._make_processor(tmp_path)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 200.0, dtype=np.float32)
        depth[40:60, 40:60] = 50.0  # vat can gan hon (nho hon ref 200 - 15)
        out = proc.mask_obstruction(frame, [(10, 10, 90, 90)], depth)
        assert (out[40:60, 40:60] == 128).all()
        assert (out[0, 0] == 0).all()
        out2 = proc.mask_obstruction(frame, [(10, 10, 90, 90)],
                                     np.full((100, 100), 200.0, dtype=np.float32))
        assert out2 is frame  # khong co vat can -> khong copy

    def test_disabled_when_no_file(self, tmp_path):
        proc = self._make_processor(tmp_path)
        assert proc.open("khong_ton_tai") is False

    def test_open_reads_paths_depth_dir(self, tmp_path):
        """depth_dir doc tu paths.depth_dir; PKU uu tien <video>-depth.avi;
        read(to_meters=True) nhan gray_to_m."""
        import cv2
        from src.depth_processor import DepthProcessor
        depth_dir = tmp_path / "depths"
        depth_dir.mkdir()
        gray_val = 10
        writer = cv2.VideoWriter(str(depth_dir / "vd-depth.avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 30, (16, 16))
        for _ in range(3):
            writer.write(np.full((16, 16, 3), gray_val, dtype=np.uint8))
        writer.release()
        cfg = make_cfg(tmp_path, **{"depth.enabled": True,
                                    "paths.depth_dir": str(depth_dir),
                                    "depth.gray_to_m": 0.256})
        proc = DepthProcessor(cfg)
        assert proc.open("vd") is True
        d = proc.read(1, to_meters=True)
        assert d is not None and d.shape == (16, 16)
        assert d[0, 0] == pytest.approx(gray_val * 0.256, rel=0.05)
        proc.close()


class TestTrackers:
    def test_iou_tracker_consistency(self):
        from src.tracker import IoUTracker
        tr = IoUTracker()
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        b1 = tr.update([PersonBox((0, 0, 10, 10), 0.9), PersonBox((50, 50, 60, 60), 0.8)], f)
        b2 = tr.update([PersonBox((51, 51, 61, 61), 0.85), PersonBox((1, 1, 11, 11), 0.95)], f)
        assert b2[0].xyxy == (1, 1, 11, 11)   # slot 0 van la nguoi ban dau

    def test_deepocsort_if_available(self):
        pytest.importorskip("boxmot")
        from src.tracker import DeepOCSORTTracker
        cfg = ConfigManager(None, overrides={"yolo.conf_threshold": 0.2})
        tr = DeepOCSORTTracker(cfg)
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        ids_seen = []
        for i in range(4):
            boxes = [PersonBox((10 + i, 10, 60 + i, 100), 0.9),
                     PersonBox((90 - i, 10, 140 - i, 100), 0.8)]
            out = tr.update(boxes, f)
            ids_seen.append([b.track_id for b in out])
        assert ids_seen[0] == ids_seen[-1]     # track id on dinh
        assert len(out) <= 2


class TestVisualizeBadFrames:
    def test_viz_bad_video(self, tmp_path):
        """Video co 4 frame loi (>gap) -> viz THEO CHUNK: folder chunk_000 chua
        video cat vung chunk + anh tung frame fail."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        from scripts.visualize_bad_frames import visualize_video
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vfail2.avi", n_frames=10)
        (tmp_path / "labels" / "vfail2.txt").write_text("3,1,10,1\n", encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg,
                                    pose_extractor=FakePose(fail_colors={0, 1, 2, 3}),
                                    yolo_detector=FakeYOLO())
        orch.process_video("vfail2")
        report = visualize_video(cfg, "vfail2")
        assert report["video_ok"] is False
        assert report["chunks"]["chunk_000"]["bad_frames"] == [1, 2, 3, 4]
        viz_dir = Path(cfg.get("paths.failed_frames_dir")) / "vfail2" / "chunk_000"
        assert len(list(viz_dir.glob("frame_*.jpg"))) == 4
        assert (viz_dir / "vfail2_chunk_000.mp4").exists()

    def test_viz_ok_video_no_output(self, tmp_path):
        """Video hoan toan ok -> khong tao viz."""
        from src.pipeline_orchestrator import PipelineOrchestrator
        from scripts.visualize_bad_frames import visualize_video
        cfg = make_cfg(tmp_path)
        _write_test_video(tmp_path / "videos" / "vok.avi", n_frames=10)
        (tmp_path / "labels" / "vok.txt").write_text("3,1,10,1\n", encoding="utf-8")
        orch = PipelineOrchestrator(cfg=cfg, pose_extractor=FakePose(),
                                    yolo_detector=FakeYOLO())
        orch.process_video("vok")
        report = visualize_video(cfg, "vok")
        assert report["video_ok"] is True
        assert report["chunks"] == {}
