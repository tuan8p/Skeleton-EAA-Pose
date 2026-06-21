"""
eaa_pose.track_io
=================
Utilities for Module 2A track timelines and status statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .datasets.base import ActionSegment


STATUS_OK = "ok"
STATUS_OUTSIDE_ACTION = "outside_action"
STATUS_NO_DETECTION = "no_detection"
STATUS_TRACK_LOST = "track_lost"
STATUS_MULTIPLE_PERSON_CANDIDATES = "multiple_person_candidates"
STATUS_READ_FAILED = "read_failed"
STATUS_UNREAD = "unread"

REVIEW_STATUSES = (
    STATUS_NO_DETECTION,
    STATUS_TRACK_LOST,
    STATUS_MULTIPLE_PERSON_CANDIDATES,
    STATUS_READ_FAILED,
)


def segment_bounds(seg: ActionSegment, total_frames: int) -> tuple[int, int] | None:
    """Return inclusive 0-based [start, end] bounds for a segment."""
    start = max(0, seg.start_frame - 1)
    end = min(seg.end_frame - 1, total_frames - 1)
    if end < start:
        return None
    return start, end


def action_frame_map(
    segments: tuple[ActionSegment, ...],
    total_frames: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return action mask and per-frame segment ids."""
    mask = np.zeros(total_frames, dtype=bool)
    seg_ids: list[list[int]] = [[] for _ in range(total_frames)]
    for seg in segments:
        bounds = segment_bounds(seg, total_frames)
        if bounds is None:
            continue
        start, end = bounds
        mask[start : end + 1] = True
        for idx in range(start, end + 1):
            seg_ids[idx].append(int(seg.seg_id))
    return mask, seg_ids


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON with UTF-8 encoding, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return out_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def track_path(out_dir: str | Path, tracks_dir: str, video_id: str) -> Path:
    """Return the track timeline path for one video."""
    return Path(out_dir) / tracks_dir / f"{video_id}_tracks.json"


def longest_status_run(frames: list[dict[str, Any]], status: str) -> int:
    """Return longest contiguous run of a status over frame records."""
    longest = 0
    current = 0
    for frame in frames:
        if frame.get("inside_action") and frame.get("status") == status:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def status_segments_and_count(
    frames: list[dict[str, Any]],
    status: str,
) -> tuple[list[int], int]:
    """Return affected segment ids and frame count for status in action frames."""
    segments: set[int] = set()
    count = 0
    for frame in frames:
        if not frame.get("inside_action"):
            continue
        if frame.get("status") != status:
            continue
        count += 1
        for seg_id in frame.get("seg_ids", []):
            segments.add(int(seg_id))
    return sorted(segments), int(count)


def summarize_track_timelines(
    timelines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build dataset-level Step 2A statistics."""
    status_counts = {STATUS_OK: 0}
    videos_by_status: dict[str, list[dict[str, Any]]] = {
        status: [] for status in REVIEW_STATUSES
    }
    num_action_frames = 0

    for timeline in timelines:
        video_id = str(timeline.get("video_id", ""))
        frames = timeline.get("frames", [])
        if not isinstance(frames, list):
            frames = []

        for frame in frames:
            if not frame.get("inside_action"):
                continue
            num_action_frames += 1
            status = str(frame.get("status", STATUS_UNREAD))
            status_counts[status] = status_counts.get(status, 0) + 1

        for status in REVIEW_STATUSES:
            segments, frame_count = status_segments_and_count(frames, status)
            if frame_count == 0:
                continue
            videos_by_status[status].append(
                {
                    "video_id": video_id,
                    "segments": segments,
                    "frame_count": frame_count,
                    "longest_gap": longest_status_run(frames, status),
                }
            )

    return {
        "num_videos": len(timelines),
        "num_action_frames": int(num_action_frames),
        "status_counts_in_action": status_counts,
        "videos_by_status": videos_by_status,
    }
