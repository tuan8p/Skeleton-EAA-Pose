"""
eaa_pose.run_track_qc_interpolate
=================================
Module 2A_QC_2 - interpolate short no_detection bbox gaps.

This step reads existing per-video track JSON files, interpolates short
``no_detection`` gaps inside action segments when nearby frames have valid
bboxes.  At action boundaries, a one-sided bbox anchor can be used when the
opposite side has no usable detection.  The step overwrites the same track JSON
files and writes
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
        max_anchor_distance = int(
            self._cfg.get("track_qc.max_interp_anchor_distance", max_gap + 2) or 0
        )
        allow_one_sided = not bool(self._cfg.get("track_qc.no_one_sided", False))
        limit = self._cfg.get("limit", None)

        track_files = sorted((out_dir / tracks_dir).glob("*_tracks.json"))
        if limit is not None:
            track_files = track_files[: max(0, int(limit))]

        print(
            f"[track-qc-interpolate] files={len(track_files)} "
            f"max_gap={max_gap} max_anchor_distance={max_anchor_distance} "
            f"allow_one_sided={allow_one_sided}"
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

            changed, n_frames = self._interpolate_timeline(
                timeline,
                max_gap,
                max_anchor_distance,
                allow_one_sided,
            )
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
        max_anchor_distance: int,
        allow_one_sided: bool,
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

            left_anchor = self._find_bbox_anchor(
                frames,
                start_idx=start - 1,
                step=-1,
                max_distance=max_anchor_distance,
            )
            right_anchor = self._find_bbox_anchor(
                frames,
                start_idx=end + 1,
                step=1,
                max_distance=max_anchor_distance,
            )
            if left_anchor is None and right_anchor is None:
                continue
            if (left_anchor is None or right_anchor is None) and not allow_one_sided:
                continue

            for frame_idx in range(start, end + 1):
                bbox = self._interpolated_bbox(
                    frame_idx=frame_idx,
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                )
                if bbox is None:
                    continue
                frame = frames[frame_idx]
                frame.setdefault("original_status", frame.get("status", STATUS_NO_DETECTION))
                frame["status"] = STATUS_INTERPOLATED_NO_DETECTION
                frame["bbox"] = [float(v) for v in bbox.tolist()]
                frame["score"] = None
                frame["track_id"] = None
                frame["num_candidates"] = 0
                changed = True
                interpolated += 1

        return changed, interpolated

    @staticmethod
    def _interpolated_bbox(
        *,
        frame_idx: int,
        left_anchor: tuple[int, np.ndarray] | None,
        right_anchor: tuple[int, np.ndarray] | None,
    ) -> np.ndarray | None:
        """Return interpolated bbox, or copy the only available anchor."""
        if left_anchor is not None and right_anchor is not None:
            left_idx, left_bbox = left_anchor
            right_idx, right_bbox = right_anchor
            denom = right_idx - left_idx
            if denom <= 0:
                return None
            alpha = (frame_idx - left_idx) / denom
            return (1.0 - alpha) * left_bbox + alpha * right_bbox

        if left_anchor is not None:
            return left_anchor[1].copy()
        if right_anchor is not None:
            return right_anchor[1].copy()
        return None

    @classmethod
    def _find_bbox_anchor(
        cls,
        frames: list[Any],
        start_idx: int,
        step: int,
        max_distance: int,
    ) -> tuple[int, np.ndarray] | None:
        """Find the nearest frame with a valid bbox.

        The immediate neighbor can be an outside-action frame without a
        detection.  In that case we skip over nearby frames until a usable bbox
        is found, bounded by ``max_distance`` to avoid bridging long unknown
        regions.
        """
        if step == 0 or max_distance <= 0:
            return None

        idx = start_idx
        checked = 0
        while 0 <= idx < len(frames) and checked < max_distance:
            bbox = cls._bbox_array(frames[idx])
            if bbox is not None:
                return idx, bbox
            idx += step
            checked += 1
        return None

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
        "--max-anchor-distance",
        dest="track_qc_max_interp_anchor_distance",
        type=int,
        default=None,
        help=(
            "Maximum number of frames to scan left/right for bbox anchors. "
            "Defaults to max_gap + 2."
        ),
    )
    p.add_argument(
        "--no-one-sided",
        dest="track_qc_no_one_sided",
        action="store_true",
        default=None,
        help=(
            "Require bbox anchors on both sides. By default, a short "
            "in-action no_detection gap may be filled from one nearby bbox "
            "anchor at action boundaries."
        ),
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
