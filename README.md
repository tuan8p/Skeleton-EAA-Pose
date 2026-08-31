# Skeleton-EAA-Pose

A two-stage, RGB-only 3D skeleton extraction pipeline that converts action video into **25-joint NTU-format skeleton sequences** without requiring depth sensors. Built as the preprocessing front-end for the **CTSK-Former** (Cross-Token Skeleton-Kinematic Transformer) action recognition model.

## Motivation

Benchmark action recognition datasets such as **PKU-MMD** and **TSU** were originally captured with depth-RGB sensor rigs (Microsoft Kinect). Their ground-truth skeleton annotations are derived from depth data and represent high-quality 3D joint positions.

In real-world deployment scenarios, depth sensors are often unavailable. This pipeline enables **RGB-only skeleton extraction** that is structurally compatible with depth-based NTU-format annotations — enabling training and evaluation of skeleton-based models on standard camera footage without hardware modification.

The primary technical challenge is the **coordinate system gap**: MediaPipe BlazePose estimates 3D world coordinates from monocular RGB using learned priors, while Kinect reconstructs 3D from depth maps. After Procrustes alignment, PA-MPJPE reduces by ~63% compared to raw MPJPE, confirming that the structural skeleton shape is accurately recovered — the dominant error is systematic offset rather than structural failure.

## Pipeline (Two Stages)

```
RGB Video ──► [Stage 1: Detection] ──► per-video .jsonl (bounding boxes)
                                               │
                                               ▼
                               [Stage 2: Pose Extraction] ──► skeleton .csv / .npy
```

**Stage 1** (`src/run_detection.py`) — YOLOv8 person detection with 3-attempt adaptive retry, IoU/DeepOCSORT tracking, and chunk-level resume.

**Stage 2** (`src/run_extraction.py`) — MediaPipe BlazePose on cropped person ROIs, 33→25 joint mapping to NTU format, temporal FN/TN classification, NaN interpolation, and bone-length normalization.

See [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md) for full pipeline behavior, design decisions, retry/fallback logic, and Mermaid workflow diagrams.

## Quick Start

```bash
# Stage 1: Detection
python src/run_detection.py --config config.yaml --dataset PKU --start 0 --end 50

# Stage 2: Pose Extraction
python src/run_extraction.py --config config.yaml --start 0 --end 50
```

Use `--max-segments N` to limit segments per video for rapid CPU-based testing.

## Output Format

| File | Description |
|------|-------------|
| `<output_dir>/<video>.csv` | Skeleton per frame: `frame_id`, `p1_x1..p1_z25`, `p2_x1..p2_z25` |
| `<detection_dir>/<video>.jsonl` | Per-frame bounding boxes `[x1,y1,x2,y2,conf]` |
| `<output_dir>/chunks/*.jsonl` | Per-frame metadata: action label, error flags, optional 2D landmarks |

Skeleton coordinates are in **world space** (meters, origin at mid-hip, camera-independent) by default — set via `output.coordinate_mode` in `config.yaml`.

## Supported Datasets

| Dataset | Videos | Persons | Annotation Format |
|---------|--------|---------|-------------------|
| **PKU-MMD** | 536 | 1–2 | `.avi` + per-action `.txt` label |
| **TSU** | — | 1 | `.mp4` + `.csv` event annotation |

## Evaluation Against Ground-Truth (PKU-MMD, 536 videos)

Ground truth is Kinect-derived 3D skeleton data. Procrustes Alignment (SVD) is applied to remove systematic coordinate system offset before structural comparison.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MPJPE | ~1.58 m | Raw 3D error — dominated by depth-space offset |
| **PA-MPJPE** | **~0.58 m** | After alignment — structural skeleton accuracy |
| PCK@0.2 | ~37% | Raw joint localization within 20% torso width |
| **PA-PCK@0.2** | **~78%** | After alignment — 78% of joints correctly placed |

PA-MPJPE is the **primary reported metric**. The 63% reduction from MPJPE→PA-MPJPE confirms that the pipeline recovers correct skeletal structure; the remaining error is due to MediaPipe's monocular depth estimation limitations.

## Requirements

- Python 3.10+
- `ultralytics`, `mediapipe>=0.10`, `opencv-python`, `numpy`
- Optional: `boxmot` for DeepOCSORT person re-identification
- Pre-trained models: `yolov26x.pt`, `models/pose_landmarker_heavy.task`

## Project Status

This pipeline is production-ready for PKU-MMD and TSU datasets. The extracted skeleton data has been validated against Kinect ground-truth (see Evaluation section) and is suitable for use as training input for skeleton-based action recognition models.

## Repository Layout

```
src/          Core pipeline: detection, pose, tracking, validation, I/O
tools/        Evaluation (MPJPE/PCK), visualization, and utility scripts
notebooks/    Kaggle-distributed and local extraction notebooks
tests/        Unit tests (pytest)
config.yaml   Single unified configuration file for all pipeline parameters
PROJECT_DESCRIPTION.md  Full technical description with workflow diagrams
```
