"""
eaa_pose.run_track_qc_interpolate
=================================
Module 2A_QC_2 - interpolate short no_detection bbox gaps.

This step reads existing per-video track JSON files, interpolates short
``no_detection`` gaps inside action segments when both neighboring frames have
valid bboxes, overwrites the same track JSON files, and writes
``track_stats_qc.json`` with the same structure as ``track_stats.json``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .track_io import (
    STATUS_INTERPOLATED_NO_DETECTION,
    STATUS_NO_DETECTION,
    read_json,
    summarize_track_timelines,
    write_json,
)


class TrackQCInterpolatePipeline:
    """Interpolate short no_detection runs in existing track timeline JSON."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg

    def run(self) -> None:
        out_dir = Path(self._cfg["out_dir"])
        tracks_dir = str(self._cfg.get("tracking.tracks_dir", "tracks"))
        stats_filename = str(
            self._cfg.get("track_qc.stats_filename", "track_stats_qc.json")
        )
        max_gap = int(self._cfg.get("track_qc.max_interp_gap", 0) or 0)
        limit = self._cfg.get("limit", None)

        track_files = sorted((out_dir / tracks_dir).glob("*_tracks.json"))
        if limit is not None:
            track_files = track_files[: max(0, int(limit))]

        print(
            f"[track-qc-interpolate] files={len(track_files)} "
            f"max_gap={max_gap}"
        )

        timelines: list[dict[str, Any]] = []
        changed_videos = 0
        interpolated_frames = 0

        for path in tqdm(
            track_files,
            desc="Interpolate track videos",
            unit="vid",
            position=0,
            dynamic_ncols=True,
        ):
            try:
                timeline = read_json(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not read track file '{path}': {exc}", stacklevel=2)
                continue

            changed, n_frames = self._interpolate_timeline(timeline, max_gap)
            if changed:
                write_json(path, timeline)
                changed_videos += 1
                interpolated_frames += n_frames
            timelines.append(timeline)

        write_json(out_dir / stats_filename, summarize_track_timelines(timelines))
        print(
            f"\nDone. Interpolated {interpolated_frames} frame(s) "
            f"in {changed_videos} video(s)."
        )
        print(f"Track QC stats -> {out_dir / stats_filename}")

    def _interpolate_timeline(
        self,
        timeline: dict[str, Any],
        max_gap: int,
    ) -> tuple[bool, int]:
        if max_gap <= 0:
            return False, 0

        frames = timeline.get("frames", [])
        if not isinstance(frames, list):
            return False, 0

        changed = False
        interpolated = 0
        idx = 0
        while idx < len(frames):
            if not self._is_interpolatable_gap_frame(frames[idx]):
                idx += 1
                continue

            start = idx
            while idx < len(frames) and self._is_interpolatable_gap_frame(frames[idx]):
                idx += 1
            end = idx - 1
            gap_len = end - start + 1

            if gap_len > max_gap:
                continue
            if start == 0 or end + 1 >= len(frames):
                continue

            left = frames[start - 1]
            right = frames[end + 1]
            left_bbox = self._bbox_array(left)
            right_bbox = self._bbox_array(right)
            if left_bbox is None or right_bbox is None:
                continue

            for offset, frame_idx in enumerate(range(start, end + 1), start=1):
                alpha = offset / (gap_len + 1)
                bbox = ((1.0 - alpha) * left_bbox + alpha * right_bbox).tolist()
                frame = frames[frame_idx]
                frame.setdefault("original_status", frame.get("status", STATUS_NO_DETECTION))
                frame["status"] = STATUS_INTERPOLATED_NO_DETECTION
                frame["bbox"] = [float(v) for v in bbox]
                frame["score"] = None
                frame["track_id"] = None
                frame["num_candidates"] = 0
                changed = True
                interpolated += 1

        return changed, interpolated

    @staticmethod
    def _is_interpolatable_gap_frame(frame: Any) -> bool:
        return (
            isinstance(frame, dict)
            and bool(frame.get("inside_action"))
            and frame.get("status") == STATUS_NO_DETECTION
        )

    @staticmethod
    def _bbox_array(frame: Any) -> np.ndarray | None:
        if not isinstance(frame, dict):
            return None
        bbox = frame.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            arr = np.asarray([float(v) for v in bbox], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(arr).all():
            return None
        return arr


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interpolate short no_detection bbox gaps in track JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    p.add_argument(
        "--max-gap",
        dest="track_qc_max_interp_gap",
        type=int,
        required=True,
        help="Maximum contiguous no_detection gap length to interpolate.",
    )
    p.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=None,
        help="Process at most N track JSON files, useful for smoke tests.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = PipelineConfig.load(args.config, cli_args=args)
    TrackQCInterpolatePipeline(cfg).run()


if __name__ == "__main__":
    main()
