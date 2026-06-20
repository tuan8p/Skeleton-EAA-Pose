"""
eaa_pose.run_pose
=================
Module 2 — RGB-to-3D-25-joint pose estimation pipeline.

Flow
----
For each video:

1. Load all action segments from the dataset adapter.
2. Open the video and process frame-by-frame:
   a. Detect persons with RTMDet.
   b. Track persons with ByteTrack (maintain identity across frames).
   c. Estimate 3-D whole-body keypoints with RTMW3D (133 pts).
   d. Map 133 → NTU-120 25 joints.
   e. Accumulate into a full-video sequence ``(T, M, 25, 6)``.
3. Apply quality control (confidence filter, gap interpolation, smoothing,
   normalization) on the full sequence.
4. Slice the QC'd sequence by each segment's [start_frame, end_frame].
5. Save coordinates as SkateFormer-style ``(3, T, 25, M)`` arrays named
   ``<out_dir>/<video_name>_act<seg_id>_<label_id>.npy``.
6. Write one training ``metadata.json`` and one QC report per video.

CLI usage
---------
    # Colab GPU (PKU v1 — after Module 1 has been run)
    python -m eaa_pose.run_pose --config configs/pku_v1.yaml --device cuda

    # Colab GPU (TSU)
    python -m eaa_pose.run_pose --config configs/tsu.yaml --device cuda

    # Local CPU smoke test (dry-run — no mmpose needed)
    python -m eaa_pose.run_pose --config configs/pku_v1.yaml \\
        --dry-run --smoke --max-videos 2 --max-frames 150 \\
        --out-dir data/samples_out
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .datasets.base import ActionSegment, VideoEntry
from .datasets.pku import PKUDataset
from .datasets.tsu import TSUDataset
from .io.sample_writer import SampleWriter
from .pose.detector import Detection, PersonDetector
from .pose.estimator_rtmw3d import PoseResult, RTMW3DEstimator
from .pose.mapping_25 import NTU25Mapper
from .pose.quality_control import SkeletonQC
from .pose.tracker import PersonTracker, TrackedPerson


# Number of channels per joint: [x, y, z, confidence, valid_mask, reconstructed_flag]
_N_CHANNELS = 6

_FRAME_OK = "ok"
_FRAME_NO_DETECTION = "no_detection"
_FRAME_TRACK_LOST = "track_lost"
_FRAME_POSE_FAILED = "pose_failed"
_FRAME_UNREAD = "unread"


# ---------------------------------------------------------------------------
# Mock classes for --dry-run (no GPU / mmpose needed)
# ---------------------------------------------------------------------------

class _MockDetector:
    """Returns a single full-frame bounding box per frame (no model needed)."""

    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        bbox = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
        return [Detection(bbox=bbox, score=self._score)]


class _MockTracker:
    """Trivial tracker: wraps every detection as track_id=0 (single person)."""

    def reset(self) -> None:
        pass

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[TrackedPerson]:
        if not detections:
            return []
        d = detections[0]
        return [TrackedPerson(track_id=0, bbox=d.bbox, score=d.score)]


class _MockEstimator:
    """Returns synthetic random 133-keypoint output (no model needed).

    Keypoint xyz values are small Gaussian noise around zero.
    Confidence values are uniformly high (0.85–1.0) to pass QC thresholds.
    """

    N_KEYPOINTS = 133

    def estimate(
        self,
        frame: np.ndarray,
        tracked_persons: list[TrackedPerson],
    ) -> list[PoseResult]:
        results: list[PoseResult] = []
        for person in tracked_persons:
            kps = np.zeros((self.N_KEYPOINTS, 4), dtype=np.float32)
            kps[:, :3] = np.random.randn(self.N_KEYPOINTS, 3).astype(np.float32) * 0.1
            kps[:, 3] = np.random.uniform(0.85, 1.0, self.N_KEYPOINTS).astype(np.float32)
            results.append(
                PoseResult(
                    track_id=person.track_id,
                    keypoints=kps,
                    bbox=person.bbox,
                )
            )
        return results


class PosePipeline:
    """Full RGB-to-3D-25-joint pose pipeline.

    Parameters
    ----------
    cfg:
        Loaded :class:`~eaa_pose.config.PipelineConfig`.  The pipeline
        reads the following config keys:

        ``detector.*``, ``tracker.*``, ``estimator.*``,
        ``quality_control.*``, ``mapping.*``, ``output.*``,
        ``video_dir``, ``segments_dir``, ``actions_xlsx``,
        ``out_dir``, ``dataset``, ``video_ext``, ``annotation``,
        ``smoke``, ``max_videos``, ``max_frames``, ``dry_run``.

    Notes
    -----
    When ``dry_run=True`` (via ``--dry-run`` CLI flag), the pipeline
    substitutes mock detector, tracker, and estimator so that the full
    pipeline flow can be tested locally without installing mmpose/mmdet.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg     = cfg
        self._dry_run = bool(cfg.get("dry_run", False))

        # Components built lazily to avoid MMPose import on CPU-only machines.
        # In dry_run mode, mock implementations are used instead.
        self._detector:  object | None = None
        self._tracker:   object | None = None
        self._estimator: object | None = None
        self._mapper:    NTU25Mapper = NTU25Mapper()
        self._qc:        SkeletonQC  = self._build_qc()
        self._writer:    SampleWriter | None = None
        self._metadata_videos: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline over the dataset."""
        cfg = self._cfg
        dataset = self._load_dataset()
        entries = dataset.load()

        mode_tag = "[dry-run]" if self._dry_run else "[gpu]"

        # Smoke-test mode: limit the number of videos processed
        is_smoke   = cfg.get("smoke", False)
        max_videos = int(cfg.get("max_videos", len(entries)) or len(entries))
        if is_smoke:
            entries = entries[:max_videos]
            print(f"{mode_tag}[smoke] Processing {len(entries)} video(s)")

        out_dir = Path(cfg["out_dir"])
        self._writer = SampleWriter(
            out_dir,
            dtype=str(cfg.get("output.dtype", "float32")),
        )
        self._metadata_videos = []

        n_saved = 0
        for entry in tqdm(entries, desc="Videos", unit="vid"):
            try:
                n_saved += self._process_video(entry)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Failed on '{entry.video_id}': {exc}", stacklevel=2)

        self._write_dataset_metadata()
        print(f"\nDone. Saved {n_saved} sample .npy files → {out_dir}")

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------

    def _process_video(self, entry: VideoEntry) -> int:
        """Process one video end-to-end. Returns number of samples saved."""
        cfg       = self._cfg
        max_frames = int(cfg.get("max_frames", 0) or 0)
        num_persons = int(cfg.get("output.num_persons", 1))

        # --- 1. Open video -----------------------------------------------
        cap = cv2.VideoCapture(str(entry.video_path))
        if not cap.isOpened():
            warnings.warn(f"Cannot open video: {entry.video_path}", stacklevel=2)
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # --- 2. Initialise per-video buffers -----------------------------
        # full_seq: (T, M, 25, 6) — filled as we read frames
        full_seq = np.zeros((total_frames, num_persons, 25, _N_CHANNELS), dtype=np.float32)
        full_seq[..., 4] = 1.0   # valid_mask default = 1 (will be corrected by QC)
        frame_statuses = [_FRAME_UNREAD for _ in range(total_frames)]

        if self._tracker is None:
            self._tracker = self._get_tracker()
        self._tracker.reset()

        # --- 3. Frame-by-frame pose extraction ---------------------------
        actual_frames = 0
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            actual_frames += 1

            h, w = frame.shape[:2]
            detections = self._get_detector().detect(frame)

            # Keep top-num_persons detections
            detections = detections[:num_persons]

            tracked = self._get_tracker().update(detections, (h, w))
            tracked = tracked[:num_persons]

            if not tracked:
                # No person found — leave this frame as zeros (invalid)
                full_seq[frame_idx, :, :, 4] = 0.0   # valid_mask = 0
                frame_statuses[frame_idx] = (
                    _FRAME_NO_DETECTION if not detections else _FRAME_TRACK_LOST
                )
                continue

            pose_results = self._get_estimator().estimate(frame, tracked)
            if not pose_results:
                full_seq[frame_idx, :, :, 4] = 0.0
                frame_statuses[frame_idx] = _FRAME_POSE_FAILED
                continue

            frame_statuses[frame_idx] = _FRAME_OK

            for person_idx, pr in enumerate(pose_results[:num_persons]):
                kp_133 = pr.keypoints           # (133, 4)
                kp_25  = self._mapper.map(kp_133)  # (25, 4)

                # Fill [x, y, z, conf] channels
                full_seq[frame_idx, person_idx, :, :4] = kp_25
                # valid_mask: 1 if confidence > 0 else 0
                full_seq[frame_idx, person_idx, :, 4] = (kp_25[:, 3] > 0).astype(np.float32)
                # reconstructed_flag: 0 (will be updated by QC)
                full_seq[frame_idx, person_idx, :, 5] = 0.0

        cap.release()

        if actual_frames == 0:
            return 0

        # Trim to actual length
        full_seq = full_seq[:actual_frames]
        frame_statuses = frame_statuses[:actual_frames]

        # --- 4. Quality control on full sequence -------------------------
        full_seq = self._qc.process(full_seq)
        self._write_video_qc_report(entry, full_seq, frame_statuses, actual_frames)

        # --- 5. Slice and save per-segment -------------------------------
        n_saved = 0
        video_samples: list[dict] = []
        for seg in entry.segments:
            bounds = self._segment_bounds(seg, actual_frames)
            if bounds is None:
                continue
            start, end = bounds
            seg_array = full_seq[start : end + 1]
            sample_path = self._writer.write(seg_array, entry.video_id, seg.seg_id, seg.label_id)
            video_samples.append(
                self._sample_metadata(
                    seg=seg,
                    array=seg_array,
                    sample_path=sample_path,
                )
            )
            n_saved += 1

        self._metadata_videos.append(
            {
                "video_id": entry.video_id,
                "num_samples": len(video_samples),
                "samples": video_samples,
            }
        )

        return n_saved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice_segment(
        self,
        full_seq: np.ndarray,
        seg: ActionSegment,
        total_frames: int,
    ) -> np.ndarray | None:
        """Extract a segment slice from the full video sequence.

        PKU/TSU annotations use 1-based frame indices.  We convert to
        0-based for array indexing.

        Parameters
        ----------
        full_seq:
            Full-video skeleton array of shape ``(T, M, 25, 6)``.
        seg:
            The action segment descriptor.
        total_frames:
            Actual number of frames in ``full_seq``.

        Returns
        -------
        Slice array of shape ``(seg_len, M, 25, 6)``, or None if the
        segment is out of bounds or empty.
        """
        bounds = self._segment_bounds(seg, total_frames)
        if bounds is None:
            return None
        start, end = bounds

        segment_array = full_seq[start : end + 1]   # (T_seg, M, 25, 6)
        if segment_array.shape[0] == 0:
            return None

        return segment_array

    @staticmethod
    def _segment_bounds(
        seg: ActionSegment,
        total_frames: int,
    ) -> tuple[int, int] | None:
        """Return inclusive 0-based [start, end] bounds for a segment."""
        start = max(0, seg.start_frame - 1)
        end = min(seg.end_frame - 1, total_frames - 1)
        if end < start:
            return None
        return start, end

    def _sample_metadata(
        self,
        seg: ActionSegment,
        array: np.ndarray,
        sample_path: Path,
    ) -> dict:
        """Build training metadata for one saved sample."""
        T = array.shape[0]
        return {
            "file": sample_path.name,
            "seg_id": int(seg.seg_id),
            "label_id": int(seg.label_id),
            "label_name": seg.label_name,
            "start_frame": int(seg.start_frame),
            "end_frame": int(seg.end_frame),
            "num_frames": int(T),
        }

    def _write_video_qc_report(
        self,
        entry: VideoEntry,
        full_seq: np.ndarray,
        frame_statuses: list[str],
        actual_frames: int,
    ) -> None:
        """Write one QC report JSON for this video."""
        if self._writer is None:
            return
        if not bool(self._cfg.get("qc_report.enabled", True)):
            return

        long_gap_threshold = int(
            self._cfg.get(
                "qc_report.long_gap_threshold",
                self._cfg.get("quality_control.max_interp_gap", 10),
            )
        )

        segment_reports: list[dict] = []
        for seg in entry.segments:
            bounds = self._segment_bounds(seg, actual_frames)
            if bounds is None:
                continue
            start, end = bounds
            seg_array = full_seq[start : end + 1]
            segment_reports.append(
                {
                    "seg_id": int(seg.seg_id),
                    "label_id": int(seg.label_id),
                    "label_name": seg.label_name,
                    "start_frame": int(seg.start_frame),
                    "end_frame": int(seg.end_frame),
                    "quality": self._qc.summarize(
                        seg_array,
                        frame_statuses=frame_statuses[start : end + 1],
                        long_gap_threshold=long_gap_threshold,
                        include_frame_statuses=bool(
                            self._cfg.get("qc_report.include_frame_statuses", False)
                        ),
                    ),
                }
            )

        metadata = {
            "video_id": entry.video_id,
            "num_frames": int(actual_frames),
            "quality": self._qc.summarize(
                full_seq,
                frame_statuses=frame_statuses,
                long_gap_threshold=long_gap_threshold,
                include_frame_statuses=bool(
                    self._cfg.get("qc_report.include_frame_statuses", False)
                ),
            ),
            "segments": segment_reports,
        }
        filename = str(self._cfg.get("qc_report.filename_template", "qc/{video_id}_qc.json"))
        filename = filename.format(video_id=entry.video_id)
        self._writer.write_metadata(metadata, filename=filename, overwrite=True)

    def _write_dataset_metadata(self) -> None:
        """Write one training metadata file for all saved samples."""
        if self._writer is None:
            return

        metadata = {
            "num_videos": len(self._metadata_videos),
            "num_samples": sum(v["num_samples"] for v in self._metadata_videos),
            "videos": self._metadata_videos,
        }
        filename = str(self._cfg.get("metadata.filename", "metadata.json"))
        self._writer.write_metadata(metadata, filename=filename, overwrite=True)

    def _load_dataset(self):
        """Instantiate the correct dataset adapter from config."""
        cfg        = self._cfg
        dataset_id = cfg.get("dataset", "pku_v1")
        video_ext  = cfg.get("video_ext", ".avi")

        if dataset_id in ("pku_v1", "pku_v2"):
            return PKUDataset(
                video_dir    = cfg["video_dir"],
                segments_dir = cfg["segments_dir"],
                actions_xlsx = cfg["actions_xlsx"],
                video_ext    = video_ext,
            )
        elif dataset_id == "tsu":
            return TSUDataset(
                video_dir    = cfg["video_dir"],
                segments_dir = cfg["segments_dir"],
                video_ext    = video_ext,
            )
        else:
            raise ValueError(f"Unknown dataset '{dataset_id}' in config.")

    def _get_detector(self) -> object:
        """Return the detector (real or mock), building it lazily on first access."""
        if self._detector is None:
            self._detector = _MockDetector() if self._dry_run else self._build_detector()
        return self._detector

    def _get_estimator(self) -> object:
        """Return the estimator (real or mock), building it lazily on first access."""
        if self._estimator is None:
            self._estimator = _MockEstimator() if self._dry_run else self._build_estimator()
        return self._estimator

    def _get_tracker(self) -> object:
        """Return the tracker (real or mock), building it lazily on first access."""
        if self._tracker is None:
            self._tracker = _MockTracker() if self._dry_run else self._build_tracker()
        return self._tracker

    def _build_detector(self) -> PersonDetector:
        cfg = self._cfg
        return PersonDetector(
            config           = cfg["detector.config"],
            checkpoint       = cfg["detector.checkpoint"],
            score_threshold  = float(cfg.get("detector.score_threshold", 0.3)),
            device           = cfg.get("detector.device", "cuda"),
        )

    def _build_tracker(self) -> PersonTracker:
        cfg = self._cfg
        return PersonTracker(
            high_thresh = float(cfg.get("tracker.high_thresh", 0.6)),
            low_thresh  = float(cfg.get("tracker.low_thresh",  0.1)),
            max_lost    = int(cfg.get("tracker.max_lost",  30)),
            min_hits    = int(cfg.get("tracker.min_hits",   3)),
        )

    def _build_estimator(self) -> RTMW3DEstimator:
        cfg = self._cfg
        # Inherit device from detector if estimator.device is not set separately.
        device = cfg.get("estimator.device", cfg.get("detector.device", "cuda"))
        return RTMW3DEstimator(
            config                 = cfg["estimator.config"],
            checkpoint             = cfg["estimator.checkpoint"],
            device                 = device,
            keypoint_conf_threshold= float(cfg.get("estimator.keypoint_conf_threshold", 0.3)),
        )

    def _build_qc(self) -> SkeletonQC:
        cfg = self._cfg
        return SkeletonQC(
            min_conf          = float(cfg.get("quality_control.min_conf",          0.3)),
            max_missing_ratio = float(cfg.get("quality_control.max_missing_ratio", 0.4)),
            max_interp_gap    = int(cfg.get("quality_control.max_interp_gap",   10)),
            smooth_window     = int(cfg.get("quality_control.smooth_window",      5)),
            smooth_polyorder  = int(cfg.get("quality_control.smooth_polyorder",   2)),
            normalize         = bool(cfg.get("quality_control.normalize",       True)),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the RGB-to-3D-25-joint pose pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True,
        help="Dataset config YAML (e.g. configs/pku_v1.yaml).",
    )
    # Optional overrides
    p.add_argument("--video-dir",    dest="video_dir",    default=None)
    p.add_argument("--segments-dir", dest="segments_dir", default=None)
    p.add_argument("--actions-xlsx", dest="actions_xlsx", default=None)
    p.add_argument("--out-dir",      dest="out_dir",      default=None)
    p.add_argument(
        "--device", dest="device", default=None,
        help="Torch device: 'cuda' or 'cpu'.",
    )
    p.add_argument(
        "--num-persons", dest="num_persons", type=int, default=None,
        help="Number of persons to track per frame.",
    )
    p.add_argument(
        "--smoke", dest="smoke", action="store_true", default=None,
        help="Smoke-test mode: process only --max-videos videos.",
    )
    p.add_argument(
        "--max-videos", dest="max_videos", type=int, default=None,
        help="(smoke) Max number of videos to process.",
    )
    p.add_argument(
        "--max-frames", dest="max_frames", type=int, default=None,
        help="(smoke) Max frames per video.",
    )
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=None,
        help=(
            "Dry-run mode: use mock detector/tracker/estimator. "
            "No mmpose/mmdet needed. Useful for local pipeline testing."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    cfg    = PipelineConfig.load(args.config, cli_args=args)
    PosePipeline(cfg).run()


if __name__ == "__main__":
    main()
