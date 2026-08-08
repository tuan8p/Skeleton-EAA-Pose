# PROJECT DESCRIPTION — Skeleton-EAA-Pose

> **Tác giả tài liệu:** Phân tích tự động từ toàn bộ mã nguồn  
> **Cập nhật:** 2026-08-07  
> **Phạm vi:** Chỉ mô tả project `Skeleton-EAA-Pose`, không đề cập các project khác trong workspace.

---

## 1. Mục tiêu tổng quát

`Skeleton-EAA-Pose` là một **pipeline trích xuất skeleton (dữ liệu bộ xương người)** từ video hành động, phục vụ cho bài toán **nhận dạng hành động (Action Recognition)**. Pipeline chuyển đổi video thô thành dữ liệu tọa độ 3D của 25 khớp xương theo chuẩn **NTU RGB+D** để làm đầu vào cho các mô hình học sâu (ví dụ: SkateFormer). Hỗ trợ hai bộ dataset:

- **PKU-MMD Phase 1** — video `.avi`, annotation `.txt`, có thể có depth `.avi`
- **TSU (Toyota Smarthome Untrimmed)** — video `.mp4`, annotation `.csv`, có thể có depth `.mp4`

---

## 2. Công nghệ & Thư viện

| Thành phần | Thư viện / Công cụ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.12 | Toàn bộ pipeline |
| Phát hiện người | `ultralytics` (YOLO26n nano) | Detect bounding box người mỗi frame |
| Tracking (2 người) | `boxmot` / DeepOCSORT + ReID | Giữ ID ổn định cho 2 người (PKU tương tác) |
| Ước lượng tư thế | `mediapipe` PoseLandmarker (Tasks API, model `.task`) | BlazePose 33 keypoints → tọa độ 3D |
| Xử lý ảnh/video | `opencv-python` (cv2) | Đọc video, crop, vẽ, lưu ảnh |
| Tính toán số | `numpy`, `scipy` | Ma trận skeleton, nội suy |
| Cấu hình | `pyyaml` | Đọc file config YAML |
| Tiến trình | `tqdm`, `psutil` | Hiển thị thanh tiến trình, giám sát tài nguyên |
| Test | `pytest` | Unit test tự động |
| Mô hình YOLO | `yolo26n.pt` (~5.5 MB) | File weights YOLO26 nano |
| Mô hình Pose | `models/pose_landmarker_full.task` (~9.4 MB) | BlazePose full model |

---

## 3. Cấu trúc thư mục

```
Skeleton-EAA-Pose/
├── config.yaml                  # Config gốc (dùng cho Colab, đường dẫn Google Drive)
├── config.local.yaml            # Config local Windows (PKU), đường dẫn thực tế
├── config.local-tsu.yaml        # Config local Windows (TSU)
├── yolo26n.pt                   # Weights YOLO26 nano (CPU-friendly)
├── requirements.txt             # Danh sách thư viện Python
│
├── models/
│   └── pose_landmarker_full.task  # Model BlazePose Tasks API
│
├── src/                         # Toàn bộ module logic OOP
│   ├── __init__.py
│   ├── action_segment.py        # Data class: 1 đoạn hành động
│   ├── annotation_reader.py     # Đọc annotation PKU (.txt) & TSU (.csv)
│   ├── config_manager.py        # Đọc/ghi YAML config theo dot-path
│   ├── demo_overlay.py          # Vẽ skeleton lên ảnh demo
│   ├── depth_processor.py       # Đọc, align depth, ghost masking, back-projection 3D
│   ├── image_preprocessor.py    # Resize letterbox, CLAHE, gamma, denoise
│   ├── logger.py                # VideoStats, PipelineLogger, CSV batch summary
│   ├── mapper.py                # Map 33 khớp BlazePose → 25 khớp NTU
│   ├── pipeline_orchestrator.py # Điều phối toàn bộ pipeline (class trung tâm)
│   ├── pose_extractor.py        # Quản lý MediaPipe PoseLandmarker
│   ├── progress_manager.py      # Resume qua progress.json
│   ├── skeleton_interpolator.py # Nội suy temporal + spatial NaN
│   ├── skeleton_saver.py        # Lưu .npy, .jsonl, ảnh lỗi
│   ├── skeleton_scaler.py       # Chuẩn hóa tỉ lệ theo cột sống
│   ├── tracker.py               # IoUTracker & DeepOCSORTTracker
│   ├── validator.py             # Validate joint confidence, phân loại lỗi
│   └── yolo_detector.py         # YOLODetector + PersonBox
│
├── scripts/
│   ├── run_extraction.py        # CLI chính để chạy pipeline skeleton extraction
│   ├── run_detection.py         # CLI chạy pipeline detection (YOLO) độc lập
│   ├── split_actions.py         # Tách chunk → file theo từng action
│   ├── merge_actions.py         # Gộp các chunk của 1 video
│   ├── visualize_bad_frames.py  # Sinh video/ảnh kiểm chứng frame lỗi
│   └── check_tsu_depth.py       # Kiểm tra tính tuyến tính depth TSU
│
├── notebooks/
│   ├── extract_skeleton.ipynb   # Notebook Colab chạy extraction
│   └── demo_overlay.ipynb       # Notebook demo vẽ skeleton lên video
│
├── tests/
│   └── test_pipeline.py         # Unit tests (pytest), ~34KB
│
└── out/                         # Thư mục output (không commit lên git)
    ├── skeletons/               # Output của pipeline skeleton
    │   ├── chunks/npy/          # Skeleton theo chunk
    │   ├── chunks/jsonl/        # Metadata mỗi frame theo chunk
    │   ├── chunks/meta/         # Segment gốc của chunk (cho split)
    │   ├── actions/npy/         # Skeleton theo từng action (sau split)
    │   ├── actions/jsonl/
    │   └── progress.json        # Resume state
    └── outputs_detect/          # Output của pipeline detection độc lập
        └── [DATASET]/
            ├── bboxes/          # Thư mục chứa jsonl kết quả
            │   ├── chunks/      # Temp jsonl cho từng chunk
            │   ├── jsonl/       # Final jsonl cho từng video
            │   └── progress.json
            ├── failed_frames/   # Lưu video MP4 và ảnh JPEG bị lỗi detect
            └── logs/            # Log text và CSV metrics```

---

## 4. Định dạng Input

### 4.1 Dataset PKU-MMD Phase 1

**Video:**
- Định dạng: `.avi`
- Thư mục: `paths.video_dir`

**Annotation:**
- Cấu trúc thư mục: Chứa trực tiếp danh sách các file annotation (vd: `0001-L.txt`, `0002-R.txt`, ...)
- Định dạng: `.txt` (CSV không có header)
- Mỗi dòng: `action_id, start_frame, end_frame, confidence`
- Ví dụ: `3, 120, 450, 1.0`
- Thư mục: `paths.annotation_dir`

**Depth (tùy chọn):**
- File: `<video>-depth.avi` hoặc `<video>-infrared.avi` hoặc `<video>.avi`
- Định dạng: gray 8-bit, mỗi pixel = byte cao của khoảng cách mm
- Chuyển đổi: `Z(m) = gray × gray_to_m` (mặc định 0.256)
- Độ phân giải: 512×424 (Kinect v2 depth), được resize về kích thước RGB

**Action IDs tương tác (2 người):** `[12, 14, 16, 18, 21, 24, 26, 27]`

### 4.2 Dataset TSU (Toyota Smarthome Untrimmed)

**Video:**
- Định dạng: `.mp4`

**Annotation:**
- Cấu trúc thư mục: Phân nhánh theo các thư mục con đại diện cho ID người (vd: `Person01/`, `Person02/`). Bên trong mỗi thư mục con là danh sách các file annotation (vd: `Person01/1.csv`). Thư viện đọc sẽ tự động quét đệ quy để gom file.
- Định dạng: `.csv` (có hoặc không có header)
- Các cột được cấu hình: `event_column`, `start_column`, `end_column`
- Cột `event` chứa **tên hành động dạng chuỗi** (không phải số ID)
- Mapping tên → ID: tự động xây dựng sorted unique → `{tên: 1..N}`, lưu vào `tsu_event_map.json`

**Depth (tùy chọn):**
- File: `<video>-depth.mp4` (320×240, gray 8-bit tương đối, **không phải mét thật**)
- Cần kiểm tra tính tuyến tính trước khi dùng back-projection (dùng `check_tsu_depth.py`)

---

## 5. Định dạng Output

### 5.1 Skeleton `.npy`

- **Shape:** `(T, 2, 25, 3)` — `float32`
  - `T` = tổng số frame trong chunk/action
  - `2` = 2 người (person 0 & person 1)
  - `25` = 25 khớp NTU
  - `3` = tọa độ (x, y, z)
- **Hệ tọa độ** (cấu hình qua `output.coordinate_mode`):
  - `"world"` (mặc định): BlazePose world 3D, đơn vị **mét**, gốc tọa độ tại mid-hip
  - `"image"`: tọa độ normalized 0-1 trên frame ảnh
  - `"camera"`: back-projection từ depth + intrinsics camera, đơn vị mét trong không gian camera
- **Chuẩn hóa tỉ lệ:** đã chia cho `mean |SpineMid - SpineBase|` theo từng người → bất biến với khoảng cách camera
- **Giá trị đặc biệt:**
  - `NaN`: khớp lỗi (chuỗi lỗi > `interpolation.max_gap` frame, chưa được nội suy)
  - `0.0`: frame TN (thực sự không có người) hoặc person vắng trong video 1 người

### 5.2 Metadata `.jsonl`

Mỗi dòng là 1 JSON object cho 1 frame:

```json
{
  "frame_id": 1234,
  "timestamp": 41.1333,
  "action_label": 3,
  "error_flags": {
    "no_detect": true
  },
  "landmarks_2d": [[0.52, 0.41], [0.53, 0.39], ...],
  "bboxes": [[120, 80, 380, 720, 0.9412]],
  "joint_conf": [[0.98, 0.95, 0.72, ...]],
  "ok": true
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `frame_id` | int | Chỉ số frame trong video (1-based) |
| `timestamp` | float | Thời gian giây = frame_id / fps |
| `action_label` | int hoặc list[int] | ID hành động; list khi frame nằm trong vùng overlap |
| `error_flags` | dict | Cờ lỗi tại lúc extract: `no_detect`, `low_conf`, `pose_fail`, `no_person`, `bbox_interpolated`, `nan_joints`, ... |
| `landmarks_2d` | list[list[float]] | Tọa độ 2D normalized của 25 khớp NTU trên frame gốc (chỉ khi `output.save_2d_in_jsonl: true`) |
| `bboxes` | list | Danh sách bbox [x1, y1, x2, y2, conf] |
| `joint_conf` | list | Confidence visibility 25 khớp theo từng người |
| `ok` | bool | Frame hợp lệ SAU nội suy (không còn NaN) |

### 5.3 Chunk Meta `.json`

```json
{
  "video_name": "0016-R",
  "chunk_index": 0,
  "segments": [
    {"action_id": 3, "start": 120, "end": 450},
    {"action_id": 5, "start": 430, "end": 700}
  ]
}
```

### 5.4 Cấu trúc đường dẫn output

```
output_dir/
├── chunks/
│   ├── npy/   <video>_chunk_000.npy, _chunk_001.npy, ...
│   ├── jsonl/ <video>_chunk_000.jsonl, ...
│   └── meta/  <video>_chunk_000.json, ...
├── actions/               (sau bước split_actions)
│   ├── npy/   <video>_000.npy, _001.npy, ...
│   └── jsonl/ <video>_000.jsonl, ...
└── progress.json
```

### 5.5 Log & Báo cáo

- **`logs/<video>.log`** — chi tiết từng video: số frame, tỉ lệ ok, lỗi, FN, TN, retry, tốc độ
- **`logs/batch_summary_<start>_<end>.csv`** — bảng tổng hợp batch với các cột:
  `video_name, total_frames, ok_frames, ok_persons, bad_frames_count, video_ok, no_detect_count, low_conf_count, pose_fail_count, retry_count, detect_ok_frames, detect_fail_frames, avg_detect_conf, fn_count, tn_count, interpolated_count, fps, avg_time_per_frame_ms, avg_conf, conf_mode`

---

## 6. Luồng xử lý chi tiết (Đã Tách Rời Detection và Pose)

Hệ thống được thiết kế thành 2 pipeline độc lập để dễ gỡ lỗi và quản lý tài nguyên.

### 6.1 Detection Pipeline (`run_detection.py`)

Nhiệm vụ: Duyệt video, dùng YOLO tìm BBox người, áp dụng Tracking (nếu cần), và Retry nếu thiếu người. Lưu kết quả ra file `jsonl`.

```
Annotation File (PKU .txt / TSU .csv)
        │
        ▼
[1] AnnotationReader → Chia Chunk (gộp overlap interval)
        ▼
[2] Quét Video (mỗi frame trong chunk):
        ├─ Có cache từ chunk trước overlap? → Lấy từ cache.
        ├─ YOLO26x Detect (conf=0.2, iou=0.7)
        ├─ Tracking (DeepOCSORT) với action tương tác.
        ├─ Đếm số bbox (valid_count). Đủ `n_expected`? → Pass.
        └─ Thiếu người? → Kích hoạt Retry:
               1. Resize imgsz=1280 + TTA
               2. Tăng sáng (+30) + CLAHE
        ▼
[3] Ghi kết quả (`jsonl`) & Phân loại lỗi
        ├─ no_detect: Không thấy ai
        ├─ low_conf: Thấy người nhưng toàn bộ dưới threshold
        └─ missing_person: Thấy người nhưng thiếu số lượng yêu cầu
        (Lưu thêm video MP4 visualize cho các chunk bị fail)
```

### 6.2 Pose Extraction Pipeline (`run_extraction.py`)

Nhiệm vụ: Đọc BBox đã sinh từ pipeline trước, crop ảnh, chạy MediaPipe Pose, mapping và lưu Skeleton.

```
Kết quả Detection (.jsonl)
        │
        ▼
[1] Phân tích Overlap & Chia Chunk
        ▼
[2] Pose Pass (đọc video)
        │ Mỗi frame → crop theo bbox có sẵn → BlazePose → 33 kp
        │ Mapper: 33 kp BlazePose → 25 khớp NTU
        │ Validator: lọc confidence, đánh NaN, phân loại lỗi
        │ → skeleton[T, 2, 25, 3] raw
        ▼
[3] Hậu xử lý
        SkeletonInterpolator  →  nội suy temporal + spatial NaN
        SkeletonScaler        →  chuẩn hóa tỉ lệ cột sống
        SkeletonSaver         →  lưu .npy + .jsonl + .json meta
        ProgressManager       →  đánh dấu chunk done
        │
        ▼
[4] Kết thúc Video
        VideoStats → viết .log + append batch_summary.csv
```

### 6.3 Xử lý rỗng (Temporal Logic trong Detection cũ)

*Ghi chú: Bước Detection nay đã được tách hoàn toàn ra pipeline riêng, Pose Pipeline chỉ đọc bbox JSONL, nhưng cấu trúc cốt lõi vẫn tận dụng lại các module dùng chung (ví dụ Ghost Masking, Validator, Interpolator).*

### 6.4 Nội suy (`SkeletonInterpolator`)

Sau khi có `skel[T, 2, 25, 3]` từ cả chunk:

**Temporal (ưu tiên):**
- Với mỗi chiều tọa độ (mỗi cột của ma trận phẳng):
  - Tìm các chuỗi NaN liên tiếp
  - Nếu chuỗi ≤ `max_gap` (3 frame):
    - Có 2 đầu: nội suy tuyến tính `np.interp`
    - Chỉ có 1 đầu: giữ nguyên giá trị đầu/cuối

**Spatial (bổ sung sau temporal):**
- Với mỗi khớp còn NaN (không thuộc chuỗi lỗi dài):
  - Dùng cây xương NTU: `SPATIAL_PARENT` dict (child → parent)
  - Offset = median(khớp_child - khớp_parent) trên các frame hợp lệ
  - Ước lượng: `child_missing = parent + offset`

**Sau nội suy:** đánh giá lại frame ok (không còn NaN trên số người kỳ vọng).

### 6.5 Chuẩn hóa tỉ lệ (`SkeletonScaler`)

```
scale_person = mean_over_t( |SpineMid_t - SpineBase_t| )
skel[:, p] /= scale_person
```

- Frame NaN và person vắng (toàn 0) không tham gia tính mean và không bị ảnh hưởng
- Mục tiêu: bất biến với khoảng cách camera

---

## 7. Mapping BlazePose (33 khớp) → NTU (25 khớp)

| ID NTU | Tên khớp | BlazePose nguồn | Cách tính |
|---|---|---|---|
| 0 | SpineBase | 23, 24 (hip_L, hip_R) | mean |
| 1 | SpineMid | 11, 12, 23, 24 | mean |
| 2 | Neck | 11, 12 (shoulder_L, R) | mean |
| 3 | Head | 0 (nose) | trực tiếp |
| 4 | ShoulderLeft | 11 | trực tiếp |
| 5 | ElbowLeft | 13 | trực tiếp |
| 6 | WristLeft | 15 | trực tiếp |
| 7 | HandLeft | 17, 19, 21 (pinky, index, thumb) | mean |
| 8 | ShoulderRight | 12 | trực tiếp |
| 9 | ElbowRight | 14 | trực tiếp |
| 10 | WristRight | 16 | trực tiếp |
| 11 | HandRight | 18, 20, 22 | mean |
| 12 | HipLeft | 23 | trực tiếp |
| 13 | KneeLeft | 25 | trực tiếp |
| 14 | AnkleLeft | 27 | trực tiếp |
| 15 | FootLeft | 31 (foot_index_L) | trực tiếp |
| 16 | HipRight | 24 | trực tiếp |
| 17 | KneeRight | 26 | trực tiếp |
| 18 | AnkleRight | 28 | trực tiếp |
| 19 | FootRight | 32 | trực tiếp |
| 20 | SpineShoulder | 11, 12 | mean |
| 21 | HandTipLeft | 19 (index_L) | trực tiếp |
| 22 | ThumbLeft | 21 | trực tiếp |
| 23 | HandTipRight | 20 (index_R) | trực tiếp |
| 24 | ThumbRight | 22 | trực tiếp |

---

## 8. Logic nghiệp vụ — Phân loại frame

### 8.1 Ba loại lỗi cứng (lúc extract)

| Mã lỗi | Điều kiện | Hành động |
|---|---|---|
| `no_detect` | YOLO không trả về box nào | Lưu ảnh frame lỗi, ghi flag |
| `low_conf` | Có box nhưng tất cả box đều có conf < threshold | Lưu ảnh frame + bbox cam |
| `missing_person` | Số lượng người thỏa mãn threshold ít hơn số lượng yêu cầu | Lưu ảnh frame lỗi |
| `pose_fail` | BlazePose trả về kết quả nhưng > 50% joint có visibility < threshold | Lưu ảnh frame lỗi |

### 8.2 Phân loại frame rỗng (Temporal Logic)

| Loại | Điều kiện | Xử lý |
|---|---|---|
| **TN** (True Negative) | Chuỗi frame rỗng ≥ `empty_run_frames` (30) | skeleton = 0, `ok = True`, không lỗi |
| **FN** (False Negative) | Chuỗi rỗng ngắn, có frame detect tốt kề bên | Retry detect + nội suy bbox, nếu vẫn rỗng ghi `fn_frames` |

### 8.3 Đánh giá frame OK (sau nội suy)

```python
frame_ok = True  nếu:
  - Là frame TN (no_person flag), HOẶC
  - Tất cả khớp của số người kỳ vọng không còn NaN
    (video 1 người: person 0; action tương tác: cả 2 người)
```

### 8.4 Xử lý vùng overlap giữa 2 action

- Dùng thuật toán **union intervals** (sweepline theo sự kiện start/end):
  - Frame nằm trong vùng chồng lặp → được extract **chỉ 1 lần**
  - `action_label` của frame đó là **list** chứa cả 2 action ID
- Khi `split_actions.py` tách về file action riêng: frame overlap được **copy vào cả 2 file action**

---

## 9. Xử lý Depth (tùy chọn)

Kích hoạt bằng `depth.enabled: true` trong config.

### 9.1 Ghost Legs Masking

- **Mục đích:** loại bỏ phần chân/thân ma (ghost legs) của người kia khi 2 người đứng gần nhau, tránh BlazePose bị lẫn lộn
- **Cơ chế tự động:**
  1. Tính `depth_ref` = median của depth tại vùng trung tâm (40%) bbox người
  2. Tìm pixel có `0 < depth < depth_ref - min_delta (15)` → vật cản gần hơn người
  3. Nếu vật cản chiếm > 1% diện tích bbox → tô màu xám `fill_color=[128,128,128]` lên frame trước khi BlazePose

### 9.2 Back-projection 3D (coordinate_mode = "camera")

Chuyển từ pixel ảnh 2D + depth → tọa độ 3D trong không gian camera:

```
X = (x_px - cx) * Z / fx
Y = (y_px - cy) * Z / fy
```

- **Intrinsics:** `fx=1059.29, fy=1059.32, cx=962.90, cy=543.40` (camera RGB Kinect v2)
- **Z lỗi** (≤ 0, mất depth): bù từ khớp cha theo cây NTU (lặp tối đa 3 lần), cuối cùng vẫn lỗi → NaN
- **PKU:** depth gray × `gray_to_m (0.256)` = Z mét thật
- **TSU:** depth gray 8-bit tương đối, không phải mét thật → khuyến nghị dùng `"world"` cho TSU

---

## 10. Cơ chế Resume

- `progress.json` được ghi tại `output_dir/progress.json`
- Cấu trúc: `{ "video_name": { "chunks_done": [0, 1, ...], "total_chunks": N, "finished": bool } }`
- Ở Detection Pipeline mới: `is_video_done` bị bỏ qua ở lớp ngoài, quét trực tiếp thư mục `chunks` để tìm sự hiện diện thực tế của file temp `.jsonl`.
- Khi chạy lại ở mức Video (tạo file merged jsonl): Skip video nếu file `.jsonl` đã tồn tại trong folder cuối.

---

## 11. Tiền xử lý ảnh (`ImagePreprocessor`)

Chạy trước khi đưa vào BlazePose để tăng chất lượng detect (cấu hình `preprocessing.*`):

| Bước | Config | Mặc định | Mô tả |
|---|---|---|---|
| Resize letterbox | `resize.enabled`, `width`, `height`, `keep_ratio` | 640×640, bật | Giữ tỉ lệ, pad đen; lưu transform (scale, pad_x, pad_y) để ánh xạ ngược tọa độ |
| Gamma correction | `gamma.enabled`, `gamma.value` | bật, 0.8 | Giảm gamma làm sáng ảnh tối |
| CLAHE | `clahe.enabled`, `clip_limit`, `tile_grid_size` | bật | Tăng tương phản cục bộ (trên kênh L của LAB) |
| Denoise | `denoise.enabled`, `denoise.method`, `denoise.strength` | bật, bilateral | Bilateral hoặc Gaussian blur |

Khi retry: luân phiên dùng ảnh đã preprocessed và ảnh thô gốc.

---

## 12. Cấu hình (config.yaml)

File YAML với cấu trúc phân cấp, truy cập qua `ConfigManager.get("section.key")`:

```yaml
dataset: PKU | TSU

mediapipe:
  model_path, num_poses, min_detection/presence/tracking_confidence, static_image_mode

yolo:
  model_name, conf_threshold, iou_threshold, max_detections, tracker

depth:
  enabled, gray_to_m, scale_to_rgb
  masking: enabled, min_delta, fill_color
  intrinsics: fx, fy, cx, cy

preprocessing:
  resize: enabled, width, height, keep_ratio
  clahe: enabled, clip_limit, tile_grid_size
  gamma: enabled, value
  denoise: enabled, method, strength

thresholds:
  confidence          # ngưỡng visibility joint (NaN nếu dưới)
  max_nan_ratio       # tỉ lệ NaN tối đa để frame không bị pose_fail

validator:
  max_retries         # số lần retry BlazePose

interpolation:
  max_gap             # chuỗi lỗi ≤ N frame thì nội suy, > N giữ NaN

temporal:
  empty_run_frames    # chuỗi rỗng ≥ N → TN
  neighbor_conf       # ngưỡng conf "tốt" để xác định FN (null = thresholds.confidence)
  bbox_interp_max_gap # chỉ nội suy bbox cho chuỗi FN ngắn hơn N

scaling:
  enabled, root_idx (SpineBase=0), spine_idx (SpineMid=1)

chunk:
  max_actions_per_chunk  # số segment tối đa mỗi chunk (mặc định 5)

pku:
  interactive_action_ids, video_ext, label_ext

tsu:
  video_ext, has_header, event_column, start_column, end_column, event_map

output:
  coordinate_mode     # world | image | camera
  save_2d_in_jsonl    # có lưu landmarks_2d vào jsonl không

paths:
  video_dir, output_dir, log_dir, failed_frames_dir, annotation_dir, depth_dir

runtime:
  max_segments_per_video  # null = full (đặt nhỏ để test)
  log_level
```

Có 3 file config:
- `config.yaml` — cài đặt Colab (đường dẫn Google Drive)
- `config.local.yaml` — cài đặt Windows local (PKU)
- `config.local-tsu.yaml` — cài đặt Windows local (TSU)

---

## 13. Scripts tiện ích

### `scripts/run_extraction.py` — CLI chính

```powershell
python scripts/run_extraction.py \
  --config config.local.yaml \
  --dataset PKU \
  --start 0 --end 10 \
  --max-segments 3
```

Argument `--dataset` và `--max-segments` override config.yaml tương ứng.

### `scripts/split_actions.py` — Tách chunk → file action

```powershell
python scripts/split_actions.py --output-dir out/skeletons --video 0016-R
```

Đọc `chunks/meta/*.json` + `chunks/npy/*.npy` + `chunks/jsonl/*.jsonl` → ghi `actions/npy/<video>_NNN.npy` cho mỗi segment. Vùng overlap được copy vào cả 2 file action.

### `scripts/merge_actions.py` — Gộp chunk thành 1 file video

```powershell
python scripts/merge_actions.py --output-dir out/skeletons --video 0016-R
```

Concatenate tất cả chunk .npy thành `<video>_all.npy`, gộp .jsonl thành `<video>_all.jsonl`.

### `scripts/visualize_bad_frames.py` — Kiểm chứng frame lỗi

```powershell
python scripts/visualize_bad_frames.py --config config.local.yaml --video 0016-R
```

Với mỗi chunk có `ok=False`:
- Sinh `failed_frames/<video>/chunk_NNN/<video>_chunk_NNN.mp4` — video cắt vùng chunk có vẽ bbox, conf, skeleton, joint conf màu xanh/đỏ
- Sinh `frame_XXXXXX.jpg` — ảnh tĩnh từng frame không ok

### `scripts/check_tsu_depth.py` — Kiểm tra tuyến tính depth TSU

Kiểm tra xem gray pixel depth TSU có tỉ lệ tuyến tính với Z thật không (qua tương quan width bbox ~ 1/Z):
- `|corr(width, gray)| > 0.7` **và** `CV(width×gray) < 0.35` → có thể dùng back-projection
- Nếu không đạt → nên giữ `output.coordinate_mode: "world"`

---

## 14. Thống kê được thu thập

| Chỉ số | Ý nghĩa |
|---|---|
| `ok_frames` | Frame **đủ khớp SAU retry + nội suy** (TN luôn ok; tương tác: đủ 2 người) |
| `ok_persons` | Số person hợp lệ lúc extract (thống kê phụ) |
| `bad_frames` | Frame **còn NaN sau nội suy** → video_ok = False |
| `error_frames` (no_detect/low_conf/pose_fail) | Lỗi cứng **lúc extract** |
| `detect_ok_frames` | Frame YOLO có ít nhất 1 bbox |
| `detect_fail_frames` | Frame YOLO không có bbox |
| `avg_detect_conf` | Conf bbox cao nhất trung bình |
| `fn_frames` | Có người bị sót (đã retry detect + nội suy bbox) |
| `tn_frames` | Thực sự không có người (skeleton = 0, không lỗi) |
| `interpolated_frames` | Frame được nội suy (bbox hoặc skeleton) |
| `retry_count` | Tổng số lần retry BlazePose |
| `avg_conf` | Mean confidence trên các frame thành công |
| `conf_mode` | Mode (giá trị phổ biến nhất) của confidence |

---

## 15. Cây phụ thuộc module

```
PipelineOrchestrator
├── ConfigManager          (đọc YAML)
├── AnnotationReader       (PKU / TSU)
│   └── ActionSegment      (data class)
├── YOLODetector           (detect bbox người)
│   └── PersonBox          (data class)
├── BasePersonTracker
│   ├── IoUTracker         (fallback, không cần thư viện)
│   └── DeepOCSORTTracker  (boxmot, ReID)
├── ImagePreprocessor      (resize, CLAHE, gamma, denoise)
├── PoseExtractor          (MediaPipe BlazePose Tasks API)
│   └── PoseResult         (data class: world_lm, image_lm, visibility)
├── Mapper                 (33 BlazePose → 25 NTU, BLAZE_TO_NTU table)
├── Validator              (confidence threshold, NaN, error classification)
│   └── FrameValidation    (data class)
├── DepthProcessor         (reader, align, ghost masking, backproject)
├── SkeletonInterpolator   (temporal linear + spatial parent-offset)
├── SkeletonScaler         (scale-by-spine normalization)
├── SkeletonSaver          (lưu .npy, .jsonl, .json, ảnh lỗi)
├── ProgressManager        (resume qua progress.json)
└── PipelineLogger         (VideoStats, .log, batch CSV)
```

---

## 16. Cài đặt và chạy (local Windows)

```powershell
# Tạo virtual env Python 3.12
py -3.12 -m venv .venv

# Cài PyTorch CPU-only
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại
.venv\Scripts\python -m pip install -r requirements.txt

# Chạy unit tests
.venv\Scripts\python -m pytest tests/ -v

# Chạy pipeline (test nhanh 3 video đầu, mỗi video 3 segment)
.venv\Scripts\python scripts\run_extraction.py \
  --config config.local.yaml \
  --dataset PKU \
  --start 0 --end 3 \
  --max-segments 3
```

---

## 17. Giới hạn và lưu ý quan trọng

1. **Không hỗ trợ GPU** — toàn bộ chạy CPU; trên Colab có thể tận dụng CPU nhiều core hơn.
2. **BlazePose chỉ detect 1 người/crop** — pipeline dùng YOLO crop từng người rồi chạy BlazePose riêng lẻ.
3. **TSU depth không phải mét thật** — file depth.mp4 của TSU là gray 8-bit lossy, không tuyến tính chắc chắn với Z mét → cần kiểm tra trước bằng `check_tsu_depth.py`.
4. **SpineShoulder (joint #20) được giữ lại** — không bị loại bỏ như trong một số pipeline khác; việc xử lý thêm (drop joint, resample 192 frame) do project `dataset-processing` đảm nhiệm.
5. **Cấm hardcode** — mọi hyperparameter đọc từ config.yaml, không viết cứng trong code (theo `.cursorrules`).
6. **Resume an toàn** — có thể dừng giữa chừng và chạy lại; pipeline bỏ qua video/chunk đã hoàn thành.
