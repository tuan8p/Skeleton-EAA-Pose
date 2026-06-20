# Skeleton-EAA-Pose

Preprocessing pipeline for the **EAA + KD + Causal Reasoning** project.
Converts raw RGB videos (PKU-MMD v1/v2, Toyota Smarthome Untrimmed) into
normalized 3-D 25-joint skeletons in NTU-120 format.

---

## Two modules

| Module | Script | Purpose |
|--------|--------|---------|
| 1 | `filter_pku_interactions` | Filter PKU-MMD v1 by highlighted rows in Actions_v2.xlsx; output filtered labels, skeletons + Actions_daily.csv |
| 2 | `run_pose` | RGB → RTMDet → ByteTrack → RTMW3D → 25-joint QC → per-sample `.npy` |

---

## Installation

### Base dependencies (Module 1 + local CPU smoke test)

```bash
pip install -r requirements.txt
```

### GPU/MMPose stack (Module 2 — Colab GPU)

Run the Colab notebook `notebooks/colab_pipeline.ipynb`, which mounts Google Drive
and installs the full stack automatically.  For a manual install:

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

### Module 2 — Run pose estimation

```bash
# PKU-MMD v1 (Colab, GPU)
python -m eaa_pose.run_pose --config configs/pku_v1.yaml --device cuda

# PKU-MMD v2
python -m eaa_pose.run_pose --config configs/pku_v2.yaml --device cuda

# TSU
python -m eaa_pose.run_pose --config configs/tsu.yaml --device cuda

# Local CPU smoke test (2 videos, 150 frames max each)
python -m eaa_pose.run_pose --config configs/pku_v1.yaml \
    --device cpu --smoke --max-videos 2 --max-frames 150 \
    --out-dir data/samples_out
```

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

---

## Config system

Each dataset has its own YAML (`configs/pku_v1.yaml`, `pku_v2.yaml`, `tsu.yaml`)
that extends `configs/base.yaml`.  Precedence: `base.yaml` < dataset YAML < CLI args.
Any explicit CLI argument overrides the matching config value.

---

## Dataset paths (Google Drive — Colab)

| Dataset | Videos | Segments |
|---------|--------|---------|
| PKU v1 | `/content/drive/MyDrive/.../RGB_VIDEO` | filtered `Label_PKUMMDv1_daily/` (Module 1 output) |
| PKU v2 | `/content/drive/MyDrive/.../RGB_VIDEO_v2` | `/content/drive/MyDrive/.../Label_PKUMMD_v2` |
| TSU | `/content/drive/MyDrive/.../Videos_mp4` | `/content/drive/MyDrive/.../Annotation_v1.0` |
