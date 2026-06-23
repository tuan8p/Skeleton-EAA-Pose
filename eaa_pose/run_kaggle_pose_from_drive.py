"""
Drive-backed Kaggle runner for Step 2B pose generation.

The normal ``run_pose`` command needs RGB videos and matching track JSON files.
This wrapper keeps Drive as durable storage, stages only a bounded batch of RGB
videos into Kaggle working storage, copies the needed track JSON files into each
worker output directory, runs pose workers on the available GPUs, syncs newly
generated ``.npy`` samples and QC reports back to Drive, then deletes local
worker outputs before the next batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .datasets.pku import PKUDataset
from .datasets.tsu import TSUDataset
from .io.sample_writer import SampleWriter
from .run_kaggle_tracks_from_drive import (
    RemoteVideo,
    _bytes_to_gb,
    _ensure_rclone,
    _list_remote_videos,
    _make_batches,
    _rclone,
    _rclone_capture,
    _rclone_path,
    _split_for_workers,
    _stage_worker_videos,
    _sync_inputs,
)
from .track_io import track_path


@dataclass(frozen=True)
class PoseVideo(RemoteVideo):
    expected_samples: tuple[str, ...]


def _try_copyto(remote_path: str, local_path: Path) -> bool:
    try:
        _rclone(["copyto", remote_path, str(local_path)])
    except subprocess.CalledProcessError:
        return False
    return True


def _list_remote_output_files(args: argparse.Namespace) -> set[str]:
    remote_out = _rclone_path(args.remote_name, args.remote_out_dir)
    _rclone(["mkdir", remote_out])
    try:
        raw = _rclone_capture(["lsf", remote_out, "--recursive", "--files-only"])
    except subprocess.CalledProcessError:
        return set()
    files = {line.strip().replace("\\", "/") for line in raw.splitlines() if line.strip()}
    files.update(Path(item).name for item in files)
    return files


def _remote_for_annotation(
    *,
    dataset_id: str,
    label_path: Path,
    remote_videos: dict[str, RemoteVideo],
) -> tuple[str, RemoteVideo | None]:
    video_id = label_path.stem
    if dataset_id == "tsu":
        subject_id = label_path.parent.name
        for key in (video_id.lower(), f"{subject_id}_{video_id}".lower()):
            if key in remote_videos:
                return video_id, remote_videos[key]
        prefix = video_id.lower()
        for key, remote_video in remote_videos.items():
            if key.startswith(prefix):
                return video_id, remote_video
        return video_id, None
    return video_id, remote_videos.get(video_id.lower())


def _expected_samples_for_label(dataset_id: str, label_path: Path) -> tuple[str, ...]:
    if dataset_id == "tsu":
        segments = TSUDataset._parse_csv(label_path)
    else:
        segments = PKUDataset._parse_label_file(label_path, {})
    return tuple(
        SampleWriter.filename(label_path.stem, int(seg.seg_id), int(seg.label_id))
        for seg in segments
    )


def _annotation_files(dataset_id: str, local_segments: Path) -> list[Path]:
    if dataset_id == "tsu":
        return sorted(local_segments.rglob("*.csv"))
    return sorted(local_segments.rglob("*.txt"))


def _pending_pose_videos(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    local_segments: Path,
    local_out: Path,
    remote_videos: dict[str, RemoteVideo],
    remote_output_files: set[str],
) -> list[PoseVideo]:
    dataset_id = str(cfg.get("dataset", "pku_v1"))
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    labels = _annotation_files(dataset_id, local_segments)

    matched: list[PoseVideo] = []
    missing_remote_video = 0
    missing_track = 0
    already_done = 0
    for label_path in labels:
        video_id, remote_video = _remote_for_annotation(
            dataset_id=dataset_id,
            label_path=label_path,
            remote_videos=remote_videos,
        )
        if remote_video is None:
            missing_remote_video += 1
            continue
        if not track_path(local_out, tracks_dir, video_id).exists():
            missing_track += 1
            continue
        expected_samples = _expected_samples_for_label(dataset_id, label_path)
        if expected_samples and all(name in remote_output_files for name in expected_samples):
            already_done += 1
            continue
        matched.append(
            PoseVideo(
                video_id=video_id,
                remote_path=remote_video.remote_path,
                basename=remote_video.basename,
                size_bytes=remote_video.size_bytes,
                expected_samples=expected_samples,
            )
        )

    matched = sorted(matched, key=lambda item: item.video_id)
    start = 0 if args.start_index is None else max(0, int(args.start_index))
    end = len(matched) if args.end_index is None else min(len(matched), int(args.end_index))
    pending = matched[start:end]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]

    print(
        "[kaggle-pose] "
        f"annotations={len(labels)} pending={len(pending)} already_done={already_done} "
        f"missing_track={missing_track} missing_remote_video={missing_remote_video} "
        f"range=[{start}:{end}]",
        flush=True,
    )
    return pending


def _copy_tracks_for_worker(
    videos: list[PoseVideo],
    *,
    cfg: PipelineConfig,
    local_out: Path,
    worker_out: Path,
) -> None:
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    dst = worker_out / tracks_dir
    dst.mkdir(parents=True, exist_ok=True)
    for video in videos:
        src = track_path(local_out, tracks_dir, video.video_id)
        if src.exists():
            shutil.copy2(src, dst / src.name)


def _worker_command(
    args: argparse.Namespace,
    *,
    config_path: Path,
    video_dir: Path,
    segments_dir: Path,
    actions_path: Path | None,
    out_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "eaa_pose.run_pose",
        "--config",
        str(config_path),
        "--video-dir",
        str(video_dir),
        "--segments-dir",
        str(segments_dir),
        "--out-dir",
        str(out_dir),
        "--device",
        "0",
    ]
    if actions_path is not None:
        cmd.extend(["--actions-xlsx", str(actions_path)])
    if args.num_persons is not None:
        cmd.extend(["--num-persons", str(args.num_persons)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _merge_metadata(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    videos: dict[str, dict[str, Any]] = {}
    for source in (base, update):
        for video in source.get("videos", []) if isinstance(source.get("videos", []), list) else []:
            if isinstance(video, dict) and video.get("video_id"):
                videos[str(video["video_id"])] = video
    payload = {
        "num_videos": len(videos),
        "num_samples": sum(int(v.get("num_samples", 0)) for v in videos.values()),
        "videos": [videos[k] for k in sorted(videos)],
    }
    return payload


def _merge_pose_stats(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    missing = set(base.get("missing_track_videos", []) or [])
    missing.update(update.get("missing_track_videos", []) or [])

    failed: dict[str, set[int]] = {}
    for source in (base, update):
        for item in source.get("pose_failed_videos", []) or []:
            if not isinstance(item, dict) or not item.get("video_id"):
                continue
            bucket = failed.setdefault(str(item["video_id"]), set())
            for seg_id in item.get("segments", []) or []:
                bucket.add(int(seg_id))

    review: dict[str, dict[str, Any]] = {}
    for source in (base, update):
        for item in source.get("videos_needing_review", []) or []:
            if not isinstance(item, dict) or not item.get("video_id"):
                continue
            video_id = str(item["video_id"])
            bucket = review.setdefault(video_id, {"video_id": video_id, "segments": set(), "reasons": set()})
            for seg_id in item.get("segments", []) or []:
                bucket["segments"].add(int(seg_id))
            for reason in item.get("reasons", []) or []:
                bucket["reasons"].add(str(reason))

    return {
        "num_videos": int(base.get("num_videos", 0)) + int(update.get("num_videos", 0)),
        "num_samples": int(base.get("num_samples", 0)) + int(update.get("num_samples", 0)),
        "missing_track_videos": sorted(missing),
        "pose_failed_videos": [
            {"video_id": video_id, "segments": sorted(segments)}
            for video_id, segments in sorted(failed.items())
        ],
        "videos_needing_review": [
            {
                "video_id": video_id,
                "segments": sorted(item["segments"]),
                "reasons": sorted(item["reasons"]),
            }
            for video_id, item in sorted(review.items())
        ],
    }


def _sync_pose_outputs(worker_out: Path, remote_out: str) -> None:
    _rclone(["copy", str(worker_out), remote_out, "--include", "*.npy", "--exclude", "*"])
    qc_dir = worker_out / "qc"
    if qc_dir.exists():
        _rclone(["copy", str(qc_dir), f"{remote_out.rstrip('/')}/qc"])


def _process_batch(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    *,
    batch_index: int,
    batch: list[PoseVideo],
    config_path: Path,
    segments_dir: Path,
    actions_path: Path | None,
    local_out: Path,
    metadata: dict[str, Any],
    pose_stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    worker_count = min(max(1, int(args.workers)), len(devices), len(batch))
    buckets = _split_for_workers(batch, worker_count)
    remote_out = _rclone_path(args.remote_name, args.remote_out_dir)

    print(
        f"[kaggle-pose] Batch {batch_index}: {len(batch)} video(s), {worker_count} worker(s)",
        flush=True,
    )

    processes: list[tuple[subprocess.Popen, Any, Path, Path]] = []
    for worker_idx, videos in enumerate(buckets):
        if not videos:
            continue
        worker_stage = Path(args.stage_dir) / f"pose_batch_{batch_index:04d}" / f"worker_{worker_idx}"
        worker_out = Path(args.work_dir) / "worker_outputs" / f"pose_batch_{batch_index:04d}_worker_{worker_idx}"
        if worker_out.exists():
            shutil.rmtree(worker_out)
        worker_out.mkdir(parents=True, exist_ok=True)
        _copy_tracks_for_worker(videos, cfg=cfg, local_out=local_out, worker_out=worker_out)
        video_dir = _stage_worker_videos(worker_stage, videos)
        cmd = _worker_command(
            args,
            config_path=config_path,
            video_dir=video_dir,
            segments_dir=segments_dir,
            actions_path=actions_path,
            out_dir=worker_out,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[worker_idx]
        log_path = Path(args.work_dir) / "logs" / f"pose_batch_{batch_index:04d}_worker_{worker_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(f"$ {' '.join(cmd)}\n\n")
        log_fh.flush()
        print(
            f"[kaggle-pose] Worker {worker_idx}: device={devices[worker_idx]} "
            f"videos={len(videos)} log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        processes.append((proc, log_fh, worker_out, worker_stage))

    failed = False
    for proc, log_fh, worker_out, worker_stage in processes:
        returncode = proc.wait()
        log_fh.close()
        worker_metadata = _read_json_if_exists(worker_out / "metadata.json")
        worker_pose_stats = _read_json_if_exists(worker_out / "pose_stats.json")
        metadata = _merge_metadata(metadata, worker_metadata)
        pose_stats = _merge_pose_stats(pose_stats, worker_pose_stats)
        _sync_pose_outputs(worker_out, remote_out)
        print(f"[kaggle-pose] Worker finished rc={returncode}", flush=True)
        if args.clean_worker_outputs and worker_out.exists():
            shutil.rmtree(worker_out)
        if args.clean_staged_videos and worker_stage.exists():
            shutil.rmtree(worker_stage)
        if returncode != 0:
            failed = True

    local_out.mkdir(parents=True, exist_ok=True)
    metadata_path = local_out / "metadata.json"
    pose_stats_path = local_out / "pose_stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pose_stats_path.write_text(json.dumps(pose_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _rclone(["copyto", str(metadata_path), f"{remote_out.rstrip('/')}/metadata.json"])
    _rclone(["copyto", str(pose_stats_path), f"{remote_out.rstrip('/')}/pose_stats.json"])

    if failed:
        raise RuntimeError(f"Pose batch {batch_index} had at least one failed worker.")
    return metadata, pose_stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Step 2B on Kaggle with Google Drive/rclone staging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--remote-name", default="gdrive")
    p.add_argument("--remote-video-dir", required=True)
    p.add_argument("--remote-segments-dir", required=True)
    p.add_argument("--remote-actions-path", default=None)
    p.add_argument("--remote-out-dir", required=True)
    p.add_argument("--work-dir", default="/kaggle/working/eaa_pose_kaggle")
    p.add_argument("--stage-dir", default="/kaggle/working/eaa_pose_kaggle/staged_videos")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batch-gb", type=float, default=15.0)
    p.add_argument("--video-ext", default=None)
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Accepted for CLI parity. The wrapper already processes all pending pose videos unless --limit is set.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-persons", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--clean-worker-outputs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clean-staged-videos", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _ensure_rclone()

    config_path = Path(args.config).resolve()
    cfg = PipelineConfig.load(config_path)
    local_segments, local_actions, local_out = _sync_inputs(args, cfg)

    remote_out = _rclone_path(args.remote_name, args.remote_out_dir)
    metadata = {}
    pose_stats = {}
    _try_copyto(f"{remote_out.rstrip('/')}/metadata.json", local_out / "metadata.remote.json")
    _try_copyto(f"{remote_out.rstrip('/')}/pose_stats.json", local_out / "pose_stats.remote.json")
    metadata = _read_json_if_exists(local_out / "metadata.remote.json")
    pose_stats = _read_json_if_exists(local_out / "pose_stats.remote.json")

    remote_videos = _list_remote_videos(args, cfg)
    remote_output_files = _list_remote_output_files(args)
    pending = _pending_pose_videos(
        args,
        cfg,
        local_segments,
        local_out,
        remote_videos,
        remote_output_files,
    )
    batches = _make_batches(
        pending,
        batch_size=max(1, int(args.batch_size)),
        max_batch_gb=max(0.1, float(args.max_batch_gb)),
    )
    print(
        f"[kaggle-pose] Plan: pending={len(pending)} batches={len(batches)} "
        f"max_batch_gb={args.max_batch_gb}",
        flush=True,
    )
    for idx, batch in enumerate(batches, start=1):
        batch_gb = _bytes_to_gb(sum(item.size_bytes for item in batch))
        print(f"[kaggle-pose] Batch {idx} planned size: {batch_gb:.2f} GB", flush=True)
        metadata, pose_stats = _process_batch(
            args,
            cfg,
            batch_index=idx,
            batch=batch,
            config_path=config_path,
            segments_dir=local_segments,
            actions_path=local_actions,
            local_out=local_out,
            metadata=metadata,
            pose_stats=pose_stats,
        )

    print("[kaggle-pose] Done.", flush=True)


if __name__ == "__main__":
    main()
