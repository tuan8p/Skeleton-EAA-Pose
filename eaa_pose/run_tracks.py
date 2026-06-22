"""
eaa_pose.run_tracks
===================
Module 2A — person-only detection/tracking timeline generation.

For each annotated video, this step runs YOLO + ByteTrack over the full
video, marks frames outside action annotations as ``outside_action``, and
writes one track timeline JSON per video plus a dataset-level
``track_stats.json``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .datasets.base import VideoEntry
from .datasets.pku import PKUDataset
from .datasets.tsu import TSUDataset
from .track_io import (
    STATUS_MULTIPLE_PERSON_CANDIDATES,
    STATUS_NO_DETECTION,
    STATUS_OK,
    STATUS_OUTSIDE_ACTION,
    STATUS_READ_FAILED,
    STATUS_TRACK_LOST,
    action_frame_map,
    read_json,
    summarize_track_timelines,
    track_path,
    write_json,
)


class _MockYOLOTracker:
    """Small deterministic tracker for --dry-run."""

    def track_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]
        return [
            {
                "bbox": [0.0, 0.0, float(w), float(h)],
                "score": 0.95,
                "track_id": 0,
            }
        ]


class _UltralyticsYOLOTracker:
    """YOLO person-only tracker wrapper."""

    def __init__(
        self,
        model_name: str,
        tracker_name: str,
        classes: list[int],
        conf: float,
        imgsz: int,
        device: str,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Ultralytics is required for run_tracks. Install with "
                "`pip install ultralytics` in Colab."
            ) from exc

        self._model = YOLO(model_name)
        self._tracker_name = tracker_name
        self._classes = classes
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def track_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self._model.track(
            source=frame,
            classes=self._classes,
            tracker=self._tracker_name,
            persist=True,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.detach().cpu().numpy()

        candidates: list[dict[str, Any]] = []
        for i, bbox in enumerate(xyxy):
            track_id = None if ids is None else int(ids[i])
            candidates.append(
                {
                    "bbox": [float(v) for v in bbox.tolist()],
                    "score": float(conf[i]),
                    "track_id": track_id,
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates


class TrackPipeline:
    """Generate person-presence/track timelines for annotated videos."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._dry_run = bool(cfg.get("dry_run", False))
        self._tracker: object | None = None

    def run(self) -> None:
        dataset = self._load_dataset()
        entries = dataset.load()
        print(f"[tracks] Loaded {len(entries)} matched video(s) from dataset adapter")

        out_dir = Path(self._cfg["out_dir"])
        tracks_dir = str(self._cfg.get("tracking.tracks_dir", "tracks"))
        stats_filename = str(self._cfg.get("tracking.stats_filename", "track_stats.json"))

        ranged_entries = self._select_index_range(entries)
        existing_timelines = self._load_existing_timelines(ranged_entries, out_dir, tracks_dir)
        pending_entries = [
            entry for entry in ranged_entries if entry.video_id not in existing_timelines
        ]
        selected_entries = self._select_entries(pending_entries)
        start_index, end_index = self._resolved_index_range(len(entries))

        print(
            f"[tracks] total={len(entries)} range=[{start_index}:{end_index}] "
            f"in_range={len(ranged_entries)} existing={len(existing_timelines)} "
            f"pending={len(pending_entries)} selected={len(selected_entries)}"
        )

        timelines_by_video = dict(existing_timelines)
        self._write_stats(out_dir, stats_filename, timelines_by_video)

        n_saved = 0
        for entry in tqdm(
            selected_entries,
            desc="Track videos",
            unit="vid",
            position=0,
            dynamic_ncols=True,
        ):
            try:
                timeline = self._process_video(entry)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Track step failed on '{entry.video_id}': {exc}", stacklevel=2)
                continue
            write_json(track_path(out_dir, tracks_dir, entry.video_id), timeline)
            timelines_by_video[entry.video_id] = timeline
            self._write_stats(out_dir, stats_filename, timelines_by_video)
            n_saved += 1

        print(f"\nDone. Saved {n_saved} new track timeline(s) -> {out_dir / tracks_dir}")
        print(f"Track stats -> {out_dir / stats_filename}")

    def _process_video(self, entry: VideoEntry) -> dict[str, Any]:
        max_frames = int(self._cfg.get("max_frames", 0) or 0)

        cap = cv2.VideoCapture(str(entry.video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {entry.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)

        action_mask, frame_seg_ids = action_frame_map(entry.segments, total_frames)
        frames: list[dict[str, Any]] = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            inside_action = bool(action_mask[frame_idx])
            seg_ids = frame_seg_ids[frame_idx]

            if not ret:
                frames.append(
                    self._frame_record(
                        frame_idx=frame_idx,
                        inside_action=inside_action,
                        seg_ids=seg_ids,
                        status=STATUS_READ_FAILED if inside_action else STATUS_OUTSIDE_ACTION,
                    )
                )
                continue

            candidates = self._get_tracker().track_frame(frame)
            selected = candidates[0] if candidates else None

            if not inside_action:
                frames.append(
                    self._frame_record(
                        frame_idx=frame_idx,
                        inside_action=False,
                        seg_ids=[],
                        status=STATUS_OUTSIDE_ACTION,
                        selected=selected,
                        num_candidates=len(candidates),
                    )
                )
                continue

            status = self._status_for_action_frame(candidates)
            frames.append(
                self._frame_record(
                    frame_idx=frame_idx,
                    inside_action=True,
                    seg_ids=seg_ids,
                    status=status,
                    selected=selected,
                    num_candidates=len(candidates),
                )
            )

        cap.release()

        return {
            "video_id": entry.video_id,
            "num_frames": len(frames),
            "frames": frames,
        }

    def _load_existing_timelines(
        self,
        entries: list[VideoEntry],
        out_dir: Path,
        tracks_dir: str,
    ) -> dict[str, dict[str, Any]]:
        """Load existing track files for dataset entries so resume is cheap."""
        timelines: dict[str, dict[str, Any]] = {}
        for entry in entries:
            path = track_path(out_dir, tracks_dir, entry.video_id)
            if not path.exists():
                continue
            try:
                timelines[entry.video_id] = read_json(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not read existing track file '{path}': {exc}", stacklevel=2)
        return timelines

    def _select_index_range(self, entries: list[VideoEntry]) -> list[VideoEntry]:
        """Sort all matched videos by video_id, then apply [start:end)."""
        sorted_entries = sorted(entries, key=lambda entry: entry.video_id)
        start_index, end_index = self._resolved_index_range(len(sorted_entries))
        return sorted_entries[start_index:end_index]

    def _resolved_index_range(self, total_entries: int) -> tuple[int, int]:
        raw_start = self._cfg.get("start_index", None)
        raw_end = self._cfg.get("end_index", None)

        start_index = 0 if raw_start is None else int(raw_start)
        end_index = total_entries if raw_end is None else int(raw_end)

        if start_index < 0:
            raise ValueError("--start-index must be >= 0")
        if raw_end is not None and end_index < start_index:
            raise ValueError("--end-index must be >= --start-index")

        return min(start_index, total_entries), min(end_index, total_entries)

    def _select_entries(self, pending_entries: list[VideoEntry]) -> list[VideoEntry]:
        """Select pending videos according to --all / --limit / smoke flags."""
        if bool(self._cfg.get("all", False)):
            return pending_entries

        limit = self._cfg.get("limit", None)
        if limit is not None:
            return pending_entries[: max(0, int(limit))]

        if bool(self._cfg.get("smoke", False)):
            max_videos = int(self._cfg.get("max_videos", len(pending_entries)) or len(pending_entries))
            return pending_entries[:max_videos]

        return pending_entries

    @staticmethod
    def _write_stats(
        out_dir: Path,
        stats_filename: str,
        timelines_by_video: dict[str, dict[str, Any]],
    ) -> None:
        timelines = [timelines_by_video[k] for k in sorted(timelines_by_video)]
        write_json(out_dir / stats_filename, summarize_track_timelines(timelines))

    @staticmethod
    def _status_for_action_frame(candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return STATUS_NO_DETECTION
        if len(candidates) > 1:
            return STATUS_MULTIPLE_PERSON_CANDIDATES
        if candidates[0].get("track_id") is None:
            return STATUS_TRACK_LOST
        return STATUS_OK

    @staticmethod
    def _frame_record(
        frame_idx: int,
        inside_action: bool,
        seg_ids: list[int],
        status: str,
        selected: dict[str, Any] | None = None,
        num_candidates: int = 0,
    ) -> dict[str, Any]:
        return {
            "frame_index": int(frame_idx),
            "inside_action": bool(inside_action),
            "seg_ids": [int(s) for s in seg_ids],
            "status": status,
            "bbox": None if selected is None else selected.get("bbox"),
            "score": None if selected is None else selected.get("score"),
            "track_id": None if selected is None else selected.get("track_id"),
            "num_candidates": int(num_candidates),
        }

    def _get_tracker(self) -> object:
        if self._tracker is not None:
            return self._tracker

        if self._dry_run:
            self._tracker = _MockYOLOTracker()
            return self._tracker

        classes = self._cfg.get("tracking.classes", [0])
        if not isinstance(classes, list):
            classes = [int(classes)]

        self._tracker = _UltralyticsYOLOTracker(
            model_name=str(self._cfg.get("tracking.model", "yolo26s.pt")),
            tracker_name=str(self._cfg.get("tracking.tracker", "bytetrack.yaml")),
            classes=[int(c) for c in classes],
            conf=float(self._cfg.get("tracking.conf", 0.25)),
            imgsz=int(self._cfg.get("tracking.imgsz", 640)),
            device=self._normalize_device(
                str(self._cfg.get("tracking.device", self._cfg.get("detector.device", "cuda")))
            ),
        )
        return self._tracker

    @staticmethod
    def _normalize_device(device: str) -> str:
        """Use Ultralytics-friendly device strings."""
        if device == "cuda":
            return "0"
        return device

    def _load_dataset(self):
        cfg = self._cfg
        dataset_id = cfg.get("dataset", "pku_v1")
        video_ext = cfg.get("video_ext", ".avi")

        if dataset_id in ("pku_v1", "pku_v2"):
            return PKUDataset(
                video_dir=cfg["video_dir"],
                segments_dir=cfg["segments_dir"],
                actions_xlsx=cfg["actions_xlsx"],
                video_ext=video_ext,
            )
        if dataset_id == "tsu":
            return TSUDataset(
                video_dir=cfg["video_dir"],
                segments_dir=cfg["segments_dir"],
                video_ext=video_ext,
            )
        raise ValueError(f"Unknown dataset '{dataset_id}' in config.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run person-only detection/tracking timeline generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--video-dir", dest="video_dir", default=None)
    p.add_argument("--segments-dir", dest="segments_dir", default=None)
    p.add_argument("--actions-xlsx", dest="actions_xlsx", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--device", dest="device", default=None)
    p.add_argument(
        "--tracking-model",
        dest="tracking_model",
        default=None,
        help="YOLO model/checkpoint for person tracking, e.g. yolo26n.pt or yolo26s.pt.",
    )
    p.add_argument(
        "--tracking-tracker",
        dest="tracking_tracker",
        default=None,
        help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml.",
    )
    p.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Process at most N pending videos that do not already have track JSON.",
    )
    p.add_argument(
        "--start-index",
        dest="start_index",
        type=int,
        default=None,
        help=(
            "Zero-based inclusive start index in the full video list sorted by "
            "video_id. Applied before existing track JSON files are skipped."
        ),
    )
    p.add_argument(
        "--end-index",
        dest="end_index",
        type=int,
        default=None,
        help=(
            "Zero-based exclusive end index in the full video list sorted by "
            "video_id. Applied before existing track JSON files are skipped."
        ),
    )
    p.add_argument(
        "--all",
        dest="all_videos",
        action="store_true",
        default=None,
        help="Process all pending videos that do not already have track JSON.",
    )
    p.add_argument("--smoke", dest="smoke", action="store_true", default=None)
    p.add_argument("--max-videos", dest="max_videos", type=int, default=None)
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = PipelineConfig.load(args.config, cli_args=args)
    TrackPipeline(cfg).run()


if __name__ == "__main__":
    main()
