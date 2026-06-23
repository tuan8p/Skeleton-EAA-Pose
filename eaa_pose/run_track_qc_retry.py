"""
eaa_pose.run_track_qc_retry
===========================
Module 2A_QC_1 - retry YOLO tracking for videos with no_detection frames.

This step scans the existing ``tracks/*_tracks.json`` files, finds videos that
currently have ``no_detection`` frames inside action segments, reruns the same
Step 2A tracking logic for those videos, overwrites their original track JSON
files, and rebuilds ``track_stats.json`` from the available track files.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import PipelineConfig
from .track_io import (
    STATUS_NO_DETECTION,
    read_json,
    summarize_track_timelines,
    status_segments_and_count,
    track_path,
    write_json,
)
from .run_tracks import TrackPipeline


class TrackQCRetryPipeline:
    """Retry Step 2A tracking for videos with no_detection in action frames."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._track_pipeline = TrackPipeline(cfg)

    def run(self) -> None:
        out_dir = Path(self._cfg["out_dir"])
        tracks_dir = str(self._cfg.get("tracking.tracks_dir", "tracks"))
        stats_filename = str(self._cfg.get("tracking.stats_filename", "track_stats.json"))
        stats_path = out_dir / stats_filename

        dataset = self._track_pipeline._load_dataset()
        entries = {entry.video_id: entry for entry in dataset.load()}
        target_ids = self._target_video_ids_from_track_files(out_dir, tracks_dir)

        limit = self._cfg.get("limit", None)
        if limit is not None:
            target_ids = target_ids[: max(0, int(limit))]

        print(
            f"[track-qc-retry] targets={len(target_ids)} "
            f"status={STATUS_NO_DETECTION}"
        )

        n_saved = 0
        missing_entries: list[str] = []
        for video_id in tqdm(
            target_ids,
            desc="Retry track videos",
            unit="vid",
            position=0,
            dynamic_ncols=True,
        ):
            entry = entries.get(video_id)
            if entry is None:
                missing_entries.append(video_id)
                continue
            try:
                timeline = self._track_pipeline._process_video(entry)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Track retry failed on '{video_id}': {exc}", stacklevel=2)
                continue
            write_json(track_path(out_dir, tracks_dir, video_id), timeline)
            n_saved += 1

        self._rebuild_stats(entries, out_dir, tracks_dir, stats_filename)

        if missing_entries:
            warnings.warn(
                f"Track retry skipped {len(missing_entries)} unknown video id(s): "
                f"{missing_entries[:5]}",
                stacklevel=2,
            )

        print(f"\nDone. Retried and overwrote {n_saved} track timeline(s).")
        print(f"Track stats -> {stats_path}")

    @classmethod
    def _target_video_ids_from_track_files(cls, out_dir: Path, tracks_dir: str) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        track_files = sorted((out_dir / tracks_dir).glob("*_tracks.json"))
        for path in track_files:
            try:
                timeline = read_json(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not read track file '{path}': {exc}", stacklevel=2)
                continue
            if not cls._has_no_detection_in_action(timeline):
                continue
            video_id = str(timeline.get("video_id") or path.name.removesuffix("_tracks.json"))
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            ids.append(video_id)
        return ids

    @staticmethod
    def _has_no_detection_in_action(timeline: dict[str, Any]) -> bool:
        frames = timeline.get("frames", [])
        if not isinstance(frames, list):
            return False
        _, frame_count = status_segments_and_count(frames, STATUS_NO_DETECTION)
        return frame_count > 0

    @staticmethod
    def _rebuild_stats(
        entries: dict[str, Any],
        out_dir: Path,
        tracks_dir: str,
        stats_filename: str,
    ) -> None:
        timelines: list[dict[str, Any]] = []
        for video_id in sorted(entries):
            path = track_path(out_dir, tracks_dir, video_id)
            if not path.exists():
                continue
            try:
                timelines.append(read_json(path))
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not read track file '{path}': {exc}", stacklevel=2)
        write_json(out_dir / stats_filename, summarize_track_timelines(timelines))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retry Step 2A tracking for videos with no_detection status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--video-dir", dest="video_dir", default=None)
    p.add_argument("--segments-dir", dest="segments_dir", default=None)
    p.add_argument("--actions-xlsx", dest="actions_xlsx", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument("--device", dest="device", default=None)
    p.add_argument("--tracking-model", dest="tracking_model", default=None)
    p.add_argument("--tracking-tracker", dest="tracking_tracker", default=None)
    p.add_argument("--limit", dest="limit", type=int, default=None)
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = PipelineConfig.load(args.config, cli_args=args)
    TrackQCRetryPipeline(cfg).run()


if __name__ == "__main__":
    main()
