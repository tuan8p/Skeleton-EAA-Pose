"""
Drive-backed Kaggle runner for Step 2A_QC_1.

This script retries videos whose synced ``tracks/*_tracks.json`` files
currently contain ``no_detection`` frames inside action segments.  It does not
trust ``track_stats.json`` for target selection.  It stages only the target
videos in bounded batches, runs retry workers on the available GPUs, merges
generated track JSON files, rebuilds ``track_stats.json`` from the local track
cache, and syncs tracks/stats back to Drive after each batch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .run_kaggle_tracks_from_drive import (
    RemoteVideo,
    _bytes_to_gb,
    _ensure_rclone,
    _list_remote_videos,
    _make_batches,
    _merge_worker_tracks,
    _split_for_workers,
    _stage_worker_videos,
    _sync_inputs,
    _write_and_sync_stats,
)
from .run_track_qc_retry import TrackQCRetryPipeline


def _target_videos(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    local_out: Path,
    remote_videos: dict[str, RemoteVideo],
) -> list[RemoteVideo]:
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    target_ids = TrackQCRetryPipeline._target_video_ids_from_track_files(
        local_out,
        tracks_dir,
    )
    if args.limit is not None:
        target_ids = target_ids[: max(0, int(args.limit))]

    targets: list[RemoteVideo] = []
    missing: list[str] = []
    for video_id in target_ids:
        remote_video = remote_videos.get(video_id.lower())
        if remote_video is None:
            missing.append(video_id)
            continue
        targets.append(
            RemoteVideo(
                video_id=video_id,
                remote_path=remote_video.remote_path,
                basename=remote_video.basename,
                size_bytes=remote_video.size_bytes,
            )
        )

    if missing:
        print(
            f"[kaggle-qc-retry] Missing remote videos for {len(missing)} target(s): {missing[:5]}",
            flush=True,
        )
    print(f"[kaggle-qc-retry] retry_targets={len(targets)}", flush=True)
    return targets


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
        "eaa_pose.run_track_qc_retry",
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
    if args.tracking_model is not None:
        cmd.extend(["--tracking-model", str(args.tracking_model)])
    if args.tracking_tracker is not None:
        cmd.extend(["--tracking-tracker", str(args.tracking_tracker)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _copy_retry_context(
    local_out: Path,
    worker_out: Path,
    cfg: PipelineConfig,
    video_ids: list[str],
) -> None:
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    stats_filename = str(cfg.get("tracking.stats_filename", "track_stats.json"))
    worker_out.mkdir(parents=True, exist_ok=True)
    (worker_out / tracks_dir).mkdir(parents=True, exist_ok=True)
    stats_path = local_out / stats_filename
    if stats_path.exists():
        shutil.copy2(stats_path, worker_out / stats_filename)
    for video_id in video_ids:
        src = local_out / tracks_dir / f"{video_id}_tracks.json"
        if src.exists():
            shutil.copy2(src, worker_out / tracks_dir / src.name)


def _process_batch(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    *,
    batch_index: int,
    batch: list[RemoteVideo],
    config_path: Path,
    segments_dir: Path,
    actions_path: Path | None,
    local_out: Path,
) -> None:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    worker_count = min(max(1, int(args.workers)), len(devices), len(batch))
    buckets = _split_for_workers(batch, worker_count)

    print(
        f"[kaggle-qc-retry] Batch {batch_index}: {len(batch)} video(s), {worker_count} worker(s)",
        flush=True,
    )

    processes: list[tuple[subprocess.Popen, Any, Path, Path]] = []
    for worker_idx, videos in enumerate(buckets):
        if not videos:
            continue
        worker_stage = Path(args.stage_dir) / f"qc_retry_batch_{batch_index:04d}" / f"worker_{worker_idx}"
        worker_out = Path(args.work_dir) / "worker_outputs" / f"qc_retry_batch_{batch_index:04d}_worker_{worker_idx}"
        if worker_out.exists():
            shutil.rmtree(worker_out)
        _copy_retry_context(local_out, worker_out, cfg, [video.video_id for video in videos])
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
        log_path = Path(args.work_dir) / "logs" / f"qc_retry_batch_{batch_index:04d}_worker_{worker_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(f"$ {' '.join(cmd)}\n\n")
        log_fh.flush()
        print(
            f"[kaggle-qc-retry] Worker {worker_idx}: device={devices[worker_idx]} "
            f"videos={len(videos)} log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        processes.append((proc, log_fh, worker_out, worker_stage))

    failed = False
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    for proc, log_fh, worker_out, worker_stage in processes:
        returncode = proc.wait()
        log_fh.close()
        copied = _merge_worker_tracks(worker_out, local_out, tracks_dir)
        print(
            f"[kaggle-qc-retry] Worker finished rc={returncode}, merged_tracks={copied}",
            flush=True,
        )
        if args.clean_worker_outputs and worker_out.exists():
            shutil.rmtree(worker_out)
        if args.clean_staged_videos and worker_stage.exists():
            shutil.rmtree(worker_stage)
        if returncode != 0:
            failed = True

    _write_and_sync_stats(args, cfg, local_out)
    if failed:
        raise RuntimeError(f"QC retry batch {batch_index} had at least one failed worker.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Step 2A_QC_1 on Kaggle with Google Drive/rclone staging.",
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batch-gb", type=float, default=15.0)
    p.add_argument("--video-ext", default=None)
    p.add_argument("--tracking-model", default=None)
    p.add_argument("--tracking-tracker", default=None)
    p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Accepted for CLI parity. QC retry already processes all no_detection targets unless --limit is set.",
    )
    p.add_argument("--limit", type=int, default=None)
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
    remote_videos = _list_remote_videos(args, cfg)
    targets = _target_videos(args, cfg, local_out, remote_videos)
    batches = _make_batches(
        targets,
        batch_size=max(1, int(args.batch_size)),
        max_batch_gb=max(0.1, float(args.max_batch_gb)),
    )
    print(
        f"[kaggle-qc-retry] Plan: targets={len(targets)} batches={len(batches)} "
        f"max_batch_gb={args.max_batch_gb}",
        flush=True,
    )
    for idx, batch in enumerate(batches, start=1):
        batch_gb = _bytes_to_gb(sum(item.size_bytes for item in batch))
        print(f"[kaggle-qc-retry] Batch {idx} planned size: {batch_gb:.2f} GB", flush=True)
        _process_batch(
            args,
            cfg,
            batch_index=idx,
            batch=batch,
            config_path=config_path,
            segments_dir=local_segments,
            actions_path=local_actions,
            local_out=local_out,
        )

    print("[kaggle-qc-retry] Done.", flush=True)


if __name__ == "__main__":
    main()
