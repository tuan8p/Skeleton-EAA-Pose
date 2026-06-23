# Skeleton-EAA-Pose

Preprocessing pipeline for the **EAA + KD + Causal Reasoning** project.
Converts raw RGB videos (PKU-MMD v1/v2, Toyota Smarthome Untrimmed) into
normalized 3-D 25-joint skeletons in NTU-120 format.

---

## Two modules

| Module | Script | Purpose |
|--------|--------|---------|
| 1 | `filter_pku_interactions` | Filter PKU-MMD v1 by highlighted rows in Actions_v2.xlsx; output filtered labels, skeletons + Actions_daily.csv |
| 2A | `run_tracks` | RGB → YOLO26s person-only detection → ByteTrack → per-video track timeline + track stats |
| 2A_QC | `run_track_qc_retry`, `run_track_qc_interpolate` | Retry no-detection videos, then interpolate short no-detection bbox gaps |
| 2B | `run_pose` | Track timeline bbox → RTMW3D → NTU-25 QC → SkateFormer `.npy` + metadata + pose stats |

---

## Installation

### Base dependencies (Module 1 + local CPU smoke test)

```bash
pip install -r requirements.txt
```

### GPU/MMPose stack (Module 2 — Colab GPU)

Run `colab_pipeline.ipynb` for Colab, or `kaggle_pipeline.ipynb` for the
Drive-backed Kaggle pipeline.  The Kaggle notebook stages RGB videos in bounded
batches under `/kaggle/working`, syncs outputs back to Drive after each batch,
and covers Step 2A, Step 2A_QC, and Step 2B.  For a manual install:

```bash
# 1. PyTorch with CUDA (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. MMPose ecosystem
pip install -U openmim
mim install mmengine mmcv mmdet mmpose

# 3. ByteTrack lap solver
pip install lap
```

---

## Quick start

### Module 1 — Filter PKU-MMD v1 daily actions

```bash
# Uses paths from configs/pku_v1.yaml (reads Actions_v2.xlsx highlighted rows)
python -m eaa_pose.filter_pku_interactions --config configs/pku_v1.yaml

# Override paths for a one-off run
python -m eaa_pose.filter_pku_interactions --config configs/pku_v1.yaml \
    --src-actions-xlsx Actions_v2.xlsx --out-actions-csv out/Actions_daily.csv
```

### Module 2A — Run person detection/tracking

```bash
# PKU-MMD v1 (Colab, GPU)
python -m eaa_pose.run_tracks --config configs/pku_v1.yaml --device cuda --all

# PKU-MMD v2
python -m eaa_pose.run_tracks --config configs/pku_v2.yaml --device cuda --all

# TSU
python -m eaa_pose.run_tracks --config configs/tsu.yaml --device cuda --all

# Process only N pending videos and compare model/tracker choices
python -m eaa_pose.run_tracks --config configs/pku_v1.yaml --device cuda \
    --limit 20 --tracking-model yolo26n.pt --tracking-tracker bytetrack.yaml

# Split work across two Colab accounts on the same output folder.
# Ranges use the full video list sorted by video_id: start inclusive, end exclusive.
python -m eaa_pose.run_tracks --config configs/pku_v1.yaml --device cuda \
    --start-index 0 --end-index 1000 --all
python -m eaa_pose.run_tracks --config configs/pku_v1.yaml --device cuda \
    --start-index 1000 --end-index 2000 --all

# Local smoke test
python -m eaa_pose.run_tracks --config configs/pku_v1.yaml \
    --device cpu --dry-run --limit 2 --max-frames 150 \
    --out-dir data/samples_out
```

`run_tracks` resumes automatically: if
`<out_dir>/tracks/<video_id>_tracks.json` already exists, that video is
skipped.  After each newly processed video, its track JSON is written
immediately.  `track_stats.json` is rebuilt from both existing and newly
generated track files.  `--start-index` and `--end-index` are applied first on
the full video list sorted by `video_id` (`start` inclusive, `end` exclusive).
After that range is selected, existing track files in the range are skipped.
`--limit N` then processes at most N remaining pending videos in that range.

### Kaggle — Drive-backed full pipeline runners

Kaggle does not mount Google Drive like Colab.  Use `rclone` to treat Drive as
the durable storage and Kaggle as a temporary GPU worker:

- existing `tracks/*.json` are synced from Drive before planning;
- Step 2A skips videos that already have `<video_id>_tracks.json`;
- Step 2A_QC_1 retries only videos listed under `no_detection`;
- Step 2A_QC_2 interpolates track JSON files without staging RGB videos;
- Step 2B skips videos whose expected `.npy` samples already exist on Drive;
- only the current batch of RGB videos is copied to `/kaggle/working`;
- two `run_tracks` workers run in parallel on `CUDA_VISIBLE_DEVICES=0` and `1`;
- generated tracks, samples, stats, and QC reports are synced back to Drive;
- staged videos are deleted before the next batch.

Recommended Kaggle setup for 2x T4 with about 19 GB working/output storage:

```bash
# Install rclone once per Kaggle session.
curl https://rclone.org/install.sh | sudo bash
rclone version
```

Configure an `rclone` remote named `gdrive` before running the pipeline.  The
simplest repeatable approach is to store a working `rclone.conf` as a Kaggle
secret or uploaded file, then copy it to `~/.config/rclone/rclone.conf`.
Kaggle notebook cells do not reliably accept interactive `rclone config` input,
so create `rclone.conf` once in Colab or a local terminal, then provide it to
Kaggle as either a private Dataset containing `rclone.conf` or a Secret named
`RCLONE_CONF`.

Example Step 2A run:

```bash
python -m eaa_pose.run_kaggle_tracks_from_drive \
    --config configs/pku_v1.yaml \
    --remote-name gdrive \
    --remote-video-dir "DACN-TN_datasets/DATN/rawdatasets/videos/PKUv1" \
    --remote-segments-dir "DACN-TN_datasets/DATN/rawdatasets/PKU_MMD_v1/Label_daily" \
    --remote-actions-path "DACN-TN_datasets/DATN/rawdatasets/Actions_daily.csv" \
    --remote-out-dir "DACN-TN_datasets/DATN/outputs/pku_v1" \
    --workers 2 \
    --devices 0,1 \
    --batch-size 8 \
    --max-batch-gb 15 \
    --tracking-model yolo26m.pt \
    --tracking-tracker bytetrack.yaml
```

Use `/kaggle/working/eaa_pose_kaggle/staged_videos` for staged videos and keep
`--max-batch-gb 15` so each copied video wave stays below about 15 GB.  The
runner deletes staged videos after each batch; the remaining working/output
space is for `tracks/*.json`, `track_stats.json`, and worker logs under
`/kaggle/working/eaa_pose_kaggle`.

The full Kaggle notebook calls these wrappers:

```bash
python -m eaa_pose.run_kaggle_tracks_from_drive ...
python -m eaa_pose.run_kaggle_track_qc_retry_from_drive ...
python -m eaa_pose.run_track_qc_interpolate ...
python -m eaa_pose.run_kaggle_pose_from_drive ...
```

### Module 2A_QC — Retry and interpolate track timelines

```bash
# 2A_QC_1: retry videos with no_detection inside action frames.
# This overwrites tracks/<video_id>_tracks.json and rebuilds track_stats.json.
python -m eaa_pose.run_track_qc_retry --config configs/pku_v1.yaml --device cuda

# 2A_QC_2: interpolate short no_detection bbox gaps.
# This overwrites tracks/<video_id>_tracks.json and writes track_stats_qc.json.
python -m eaa_pose.run_track_qc_interpolate --config configs/pku_v1.yaml \
    --max-gap 10
```

`run_track_qc_interpolate` searches for nearby bbox anchors around a short
in-action `no_detection` run.  The anchor frames do not need to be immediately
adjacent, so a neighboring `outside_action` frame without a bbox no longer
blocks interpolation.  At action boundaries, if only one nearby bbox anchor
exists, the missing in-action frame is filled from that one anchor.  Use
`--max-anchor-distance N` to control how far left/right it may scan; omitted
means `max_gap + 2`.  Use `--no-one-sided` to require anchors on both sides.

Interpolated frames keep the original status for review:

```json
{
  "status": "interpolated_no_detection",
  "original_status": "no_detection"
}
```

### Module 2B — Run pose estimation

```bash
# PKU-MMD v1 (Colab, GPU)
python -m eaa_pose.run_pose --config configs/pku_v1.yaml --device cuda

# PKU-MMD v2
python -m eaa_pose.run_pose --config configs/pku_v2.yaml --device cuda

# TSU
python -m eaa_pose.run_pose --config configs/tsu.yaml --device cuda

# Local CPU smoke test (2 videos, 150 frames max each)
python -m eaa_pose.run_pose --config configs/pku_v1.yaml \
    --device cpu --dry-run --smoke --max-videos 2 --max-frames 150 \
    --out-dir data/samples_out
```

`run_pose` expects matching track files from `run_tracks` in
`<out_dir>/tracks/`.

---

## Output format

Each action segment is saved as:

```
<out_dir>/<video_name>_act<seg_id>_<label_id>.npy
```

Array shape: `[3, T, 25, M]`

| Axis | Meaning |
|------|---------|
| 3 | coordinate channels `[x, y, z]` |
| T | number of frames in the segment |
| 25 | joints in NTU-120 / PKU-MMD Kinect v2 order |
| M | number of persons (default 1) |

The output directory also contains one dataset-level metadata file:

```
<out_dir>/metadata.json
```

This file groups generated samples by `video_id` and stores labels plus frame
ranges for EAA training. QC fields are not exposed in the training metadata.

Each processed video also gets a separate QC report:

```
<out_dir>/qc/<video_id>_qc.json
```

Module 2A also writes:

```
<out_dir>/tracks/<video_id>_tracks.json
<out_dir>/track_stats.json
<out_dir>/track_stats_qc.json
```

`track_stats.json` lists each non-`ok` status inside action segments by
status (`no_detection`, `track_lost`, `multiple_person_candidates`,
`read_failed`) so difficult videos can be reviewed quickly.
`track_stats_qc.json` has the same schema after 2A_QC_2 and may also include
`interpolated_no_detection`.

Track timeline JSON stores only the fields needed by pose/review:
`video_id`, `num_frames`, and per-frame `frame_index`, `inside_action`,
`seg_ids`, `status`, `bbox`, `score`, `track_id`, `num_candidates`.

Module 2B also writes:

```
<out_dir>/pose_stats.json
```

`pose_stats.json` lists missing track files, pose failures, and videos/segments
that need QC review.

---

## Config system

Each dataset has its own YAML (`configs/pku_v1.yaml`, `pku_v2.yaml`, `tsu.yaml`)
that extends `configs/base.yaml`.  Precedence: `base.yaml` < dataset YAML < CLI args.
Any explicit CLI argument overrides the matching config value.

---

## Dataset paths (Google Drive — Colab)

| Dataset | Videos | Segments |
|---------|--------|---------|
| PKU v1 | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/videos/PKUv1` | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/PKU_MMD_v1/Label_daily` |
| PKU v2 | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/PKU_MMD_v2/RGB` | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/PKU_MMD_v2/Label` |
| TSU | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/TSU/mp4` | `/content/drive/MyDrive/ĐACN-TN_datasets/ĐATN/rawdatasets/TSU/Annotation_v1.0` |
