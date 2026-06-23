"""
eaa_pose.run_kaggle_tracks_from_drive
=====================================
Kaggle orchestration wrapper for Module 2A.

This runner keeps Google Drive as the durable store and uses Kaggle only as a
temporary GPU worker:

1. Sync lightweight annotations and existing track JSON files from Drive.
2. Build a pending video list by checking which ``*_tracks.json`` files exist.
3. Stage only a small batch of videos into ``/kaggle/working``.
4. Run ``run_tracks`` in parallel workers, one worker per GPU.
5. Merge generated tracks locally, rebuild stats, and sync tracks back to Drive.
6. Delete staged videos before the next batch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .track_io import read_json, summarize_track_timelines, write_json


@dataclass(frozen=True)
class RemoteVideo:
    video_id: str
    remote_path: str
    basename: str
    size_bytes: int


def _run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"[kaggle-drive] $ {printable}", flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {printable}\n\n")
        fh.flush()
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)


def _rclone_path(remote_name: str, path: str) -> str:
    if ":" in path.split("/", 1)[0]:
        return path
    return f"{remote_name}:{path.strip('/')}"


def _rclone(args: list[str]) -> None:
    _run(["rclone", *args])


def _rclone_capture(args: list[str]) -> str:
    cmd = ["rclone", *args]
    print(f"[kaggle-drive] $ {' '.join(cmd)}", flush=True)
    return subprocess.check_output(cmd, text=True, encoding="utf-8")


def _ensure_rclone() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is not installed. In Kaggle, install it first, for example: "
            "`curl https://rclone.org/install.sh | sudo bash`."
        )


def _sync_inputs(args: argparse.Namespace, cfg: PipelineConfig) -> tuple[Path, Path | None, Path]:
    work_dir = Path(args.work_dir)
    local_segments = work_dir / "inputs" / "segments"
    local_out = work_dir / "outputs"
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))

    local_segments.mkdir(parents=True, exist_ok=True)
    (local_out / tracks_dir).mkdir(parents=True, exist_ok=True)

    remote_segments = _rclone_path(args.remote_name, args.remote_segments_dir)
    remote_out = _rclone_path(args.remote_name, args.remote_out_dir)

    print("[kaggle-drive] Sync annotation files from Drive", flush=True)
    _rclone(
        [
            "copy",
            remote_segments,
            str(local_segments),
            "--include",
            "*.txt",
            "--include",
            "*.csv",
        ]
    )

    local_actions: Path | None = None
    if args.remote_actions_path:
        local_actions = work_dir / "inputs" / Path(args.remote_actions_path).name
        remote_actions = _rclone_path(args.remote_name, args.remote_actions_path)
        _rclone(["copyto", remote_actions, str(local_actions)])

    print("[kaggle-drive] Sync existing track JSON files from Drive", flush=True)
    _rclone(["mkdir", f"{remote_out.rstrip('/')}/{tracks_dir}"])
    _rclone(
        [
            "copy",
            f"{remote_out.rstrip('/')}/{tracks_dir}",
            str(local_out / tracks_dir),
            "--include",
            "*_tracks.json",
        ]
    )

    return local_segments, local_actions, local_out


def _list_remote_videos(args: argparse.Namespace, cfg: PipelineConfig) -> dict[str, RemoteVideo]:
    remote_video_dir = _rclone_path(args.remote_name, args.remote_video_dir)
    video_ext = str(args.video_ext or cfg.get("video_ext", ".avi"))
    ext_lower = video_ext.lower()

    # Format "ps" means path + size. rclone prints tab-separated columns.
    raw = _rclone_capture(["lsf", remote_video_dir, "--recursive", "--files-only", "--format", "ps"])
    lookup: dict[str, RemoteVideo] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(";", 1)
        if len(parts) != 2:
            parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        rel_path, size_raw = parts[0], parts[1]
        if not rel_path.lower().endswith(ext_lower):
            continue
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = 0

        basename = Path(rel_path).name
        stem = Path(rel_path).stem.lower()
        remote_path = f"{remote_video_dir.rstrip('/')}/{rel_path}"
        item = RemoteVideo(
            video_id=Path(rel_path).stem,
            remote_path=remote_path,
            basename=basename,
            size_bytes=size_bytes,
        )
        lookup.setdefault(stem, item)
        if stem.endswith("_color"):
            lookup.setdefault(stem[: -len("_color")], item)
        else:
            lookup.setdefault(f"{stem}_color", item)
    return lookup


def _load_pending_videos(
    args: argparse.Namespace,
    cfg: PipelineConfig,
    local_segments: Path,
    local_out: Path,
    remote_videos: dict[str, RemoteVideo],
) -> list[RemoteVideo]:
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    label_files = sorted(local_segments.rglob("*.txt"))

    matched: list[RemoteVideo] = []
    missing = 0
    for label_path in label_files:
        video_id = label_path.stem
        remote_video = remote_videos.get(video_id.lower())
        if remote_video is None:
            missing += 1
            continue
        matched.append(
            RemoteVideo(
                video_id=video_id,
                remote_path=remote_video.remote_path,
                basename=remote_video.basename,
                size_bytes=remote_video.size_bytes,
            )
        )

    matched = sorted(matched, key=lambda item: item.video_id)
    start = 0 if args.start_index is None else max(0, int(args.start_index))
    end = len(matched) if args.end_index is None else min(len(matched), int(args.end_index))
    in_range = matched[start:end]

    pending = [
        item
        for item in in_range
        if not (local_out / tracks_dir / f"{item.video_id}_tracks.json").exists()
    ]
    if args.limit is not None:
        pending = pending[: max(0, int(args.limit))]

    print(
        "[kaggle-drive] "
        f"labels={len(label_files)} matched={len(matched)} missing_videos={missing} "
        f"range=[{start}:{end}] existing={len(in_range) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    return pending


def _make_batches(
    videos: list[RemoteVideo],
    *,
    batch_size: int,
    max_batch_gb: float,
) -> list[list[RemoteVideo]]:
    max_bytes = int(max_batch_gb * 1024 * 1024 * 1024)
    batches: list[list[RemoteVideo]] = []
    cur: list[RemoteVideo] = []
    cur_size = 0
    for video in videos:
        would_count = len(cur) >= batch_size
        would_size = cur and cur_size + video.size_bytes > max_bytes
        if would_count or would_size:
            batches.append(cur)
            cur = []
            cur_size = 0
        cur.append(video)
        cur_size += video.size_bytes
    if cur:
        batches.append(cur)
    return batches


def _bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / (1024 * 1024 * 1024)


def _split_for_workers(videos: list[RemoteVideo], worker_count: int) -> list[list[RemoteVideo]]:
    buckets: list[list[RemoteVideo]] = [[] for _ in range(worker_count)]
    sizes = [0 for _ in range(worker_count)]
    for video in sorted(videos, key=lambda item: item.size_bytes, reverse=True):
        idx = min(range(worker_count), key=lambda i: sizes[i])
        buckets[idx].append(video)
        sizes[idx] += video.size_bytes
    return buckets


def _stage_worker_videos(worker_stage: Path, videos: list[RemoteVideo]) -> Path:
    video_dir = worker_stage / "videos"
    if worker_stage.exists():
        shutil.rmtree(worker_stage)
    video_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        _rclone(["copyto", video.remote_path, str(video_dir / video.basename)])
    return video_dir


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
        "eaa_pose.run_tracks",
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
        "--all",
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


def _merge_worker_tracks(worker_out: Path, local_out: Path, tracks_dir: str) -> int:
    src = worker_out / tracks_dir
    dst = local_out / tracks_dir
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return 0
    copied = 0
    for path in sorted(src.glob("*_tracks.json")):
        shutil.copy2(path, dst / path.name)
        copied += 1
    return copied


def _write_and_sync_stats(args: argparse.Namespace, cfg: PipelineConfig, local_out: Path) -> None:
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))
    stats_filename = str(cfg.get("tracking.stats_filename", "track_stats.json"))
    timelines: list[dict[str, Any]] = []
    for path in sorted((local_out / tracks_dir).glob("*_tracks.json")):
        try:
            timelines.append(read_json(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[kaggle-drive] Warning: cannot read track JSON {path}: {exc}", flush=True)
    stats_path = local_out / stats_filename
    write_json(stats_path, summarize_track_timelines(timelines))

    remote_out = _rclone_path(args.remote_name, args.remote_out_dir)
    _rclone(["copy", str(local_out / tracks_dir), f"{remote_out.rstrip('/')}/{tracks_dir}"])
    _rclone(["copyto", str(stats_path), f"{remote_out.rstrip('/')}/{stats_filename}"])


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
    tracks_dir = str(cfg.get("tracking.tracks_dir", "tracks"))

    print(
        f"[kaggle-drive] Batch {batch_index}: {len(batch)} video(s), {worker_count} worker(s)",
        flush=True,
    )

    processes: list[tuple[subprocess.Popen, Any, Path, Path]] = []
    for worker_idx, videos in enumerate(buckets):
        if not videos:
            continue
        worker_stage = Path(args.stage_dir) / f"batch_{batch_index:04d}" / f"worker_{worker_idx}"
        worker_out = Path(args.work_dir) / "worker_outputs" / f"batch_{batch_index:04d}_worker_{worker_idx}"
        if worker_out.exists():
            shutil.rmtree(worker_out)
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
        log_path = Path(args.work_dir) / "logs" / f"batch_{batch_index:04d}_worker_{worker_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[kaggle-drive] Worker {worker_idx}: device={devices[worker_idx]} "
            f"videos={len(videos)} log={log_path}",
            flush=True,
        )
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(f"$ {' '.join(cmd)}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        processes.append((proc, log_fh, worker_out, worker_stage))

    failed = False
    for proc, log_fh, worker_out, worker_stage in processes:
        returncode = proc.wait()
        log_fh.close()
        copied = _merge_worker_tracks(worker_out, local_out, tracks_dir)
        print(
            f"[kaggle-drive] Worker finished rc={returncode}, merged_tracks={copied}",
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
        raise RuntimeError(f"Batch {batch_index} had at least one failed worker. See logs.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Step 2A on Kaggle with Google Drive/rclone staging.",
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
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)
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
    pending = _load_pending_videos(args, cfg, local_segments, local_out, remote_videos)
    batches = _make_batches(
        pending,
        batch_size=max(1, int(args.batch_size)),
        max_batch_gb=max(0.1, float(args.max_batch_gb)),
    )

    print(
        f"[kaggle-drive] Plan: pending={len(pending)} batches={len(batches)} "
        f"batch_size={args.batch_size} max_batch_gb={args.max_batch_gb}",
        flush=True,
    )
    for idx, batch in enumerate(batches, start=1):
        batch_bytes = sum(item.size_bytes for item in batch)
        if _bytes_to_gb(batch_bytes) > float(args.max_batch_gb):
            print(
                f"[kaggle-drive] Warning: batch {idx} is {_bytes_to_gb(batch_bytes):.2f} GB, "
                "above max_batch_gb because at least one video is oversized.",
                flush=True,
            )
        else:
            print(
                f"[kaggle-drive] Batch {idx} planned size: {_bytes_to_gb(batch_bytes):.2f} GB "
                f"({len(batch)} video(s))",
                flush=True,
            )
    for idx, batch in enumerate(batches, start=1):
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

    print("[kaggle-drive] Done.", flush=True)


if __name__ == "__main__":
    main()
