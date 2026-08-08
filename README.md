# Skeleton-EAA-Pose

Pipeline trich xuat skeleton tu video (PKU-MMD Phase 1 / TSU): YOLO26n detect person
cho moi frame, DeepOCSORT cho action tuong tac 2 nguoi (PKU), BlazePose tren crop,
Temporal Logic phan biet "that su khong nguoi" (TN) vs "co nguoi bi sot" (FN),
retry, noi suy temporal/spatial, chuan hoa ty le (scale-by-spine), chunking, resume,
log chi tiet, demo overlay, ho tro Depth (ghost legs masking + back-projection 3D
goc camera). Chi tiet thiet ke: `PLAN.md`.

## Luong xu ly (Flow)

```
Annotation (PKU txt / TSU csv)
   |
   v
[1] AnnotationReader -> list ActionSegment (video_path, action_id, start, end)
   |        TSU: event ten -> auto event_map (sorted unique -> id, luu tsu_event_map.json)
   v
[2] Overlap DUNG CHUNG: vung giao giua 2 action chi extract 1 lan (union
   |   intervals); action_label = list tai vung giao. Moi action giu nguyen
   |   [start, end] -> split_actions ghi vung overlap vao CA 2 file action.
   v
[3] Chia chunk: toi da `chunk.max_actions_per_chunk` (=5) segment/chunk
   v
[4] Moi chunk = 2 PHA (skip neu progress.json da ghi done + file ton tai):
   |
   |   PHA 1 (detect-only, doc video 1 luot):
   |   - Moi frame qua YOLO26n. Action tuong tac PKU (12,14,16,18,21,24,26,27):
   |     DeepOCSORT giu slot person1/2 on dinh; con lai: top-1 box (single person).
   |   - Temporal Logic: frame trong (khong box conf >= neighbor_conf):
   |       * chuoi trong >= `temporal.empty_run_frames` (30) -> TN: thuc su khong
   |         nguoi -> skeleton = 0, KHONG loi, khong noi suy.
   |       * chuoi ngan giua 2 lan can co detect -> FN: co nguoi bi sot ->
   |         noi suy bbox tuyen tinh theo thoi gian (<= bbox_interp_max_gap).
   |
   |   PHA 2 (pose, doc lai video):
   |   - Frame TN -> skeleton 0. Frame co box: crop tung person
   |     -> [depth.enabled: ghost legs masking auto neu phat hien vat can]
   |     -> BlazePose tren crop. Frame FN: retry detect toi da 2 lan (bien the
   |     anh) truoc khi dung bbox noi suy.
   |   - Mapper: 33 khop BlazePose -> 25 khop NTU (mean cac khop thanh phan).
   |   - Validator PER JOINT: joint conf >= `thresholds.confidence` -> giu;
   |     < threshold -> NaN. Retry pose toi da `validator.max_retries` (=2).
   |     Loi cung: no_detect / low_conf / pose_fail -> anh loi failed_frames/.
   |   - Toa do (`output.coordinate_mode`): "world" (BlazePose 3D met, goc
   |     mid-hip - mac dinh) | "image" | "camera" (back-projection pixel +
   |     depth*gray_to_m + intrinsics; Z loi -> bu tu khop cha; thieu depth ->
   |     fallback world + warning).
   |   - Metadata frame: frame_id, timestamp, action_label (int|list),
   |     error_flags, landmarks_2d, bboxes, joint_conf
   v
[5] Cuoi chunk: SkeletonInterpolator (temporal <= `interpolation.max_gap`=3;
   |   spatial bu tu khop cha; chuoi > 3 -> giu NaN; tra filled mask)
   |   -> SkeletonScaler: chia mean |SpineMid - SpineBase| tung person
   |   -> Danh gia ok SAU noi suy: frame ok = khong con NaN tren person ky vong
   |      (TN luon ok; tuong tac: du 2 person) -> ghi "ok" vao tung dong jsonl
   |   -> Luu chunks/{npy,jsonl,meta}/<video>_chunk_NNN.* (meta: segments goc)
   |   -> progress.json: mark_chunk_done (resume bo qua chunk da xong)
   v
[6] Cuoi video: <video>.log + batch_summary_<start>_<end>.csv:
    ok_frames (sau noi suy), ok_persons (luc extract), bad_frames, video_ok,
    detect_ok/fail_frames + avg_detect_conf (YOLO), fn_frames, tn_frames,
    interpolated_frames, loi extract 3 loai, retry, fps, avg_conf, conf_mode
```

## Thong ke (cap frame)

- `ok_frames`: so frame **du joints SAU retry + noi suy** (action tuong tac: du 2x25
  joints; frame TN luon ok).
- `ok_persons`: so person hop le luc extract (thong ke phu).
- `error_frames` (no_detect/low_conf/pose_fail): loi cung LUC EXTRACT.
- `detect_ok_frames` / `detect_fail_frames` / `avg_detect_conf`: theo YOLO.
- `fn_frames`: co nguoi bi sot (da retry detect + noi suy bbox).
- `tn_frames`: thuc su khong co nguoi (skeleton = 0, khong loi).
- `interpolated_frames`: frame duoc noi suy (bbox hoac skeleton).
- `bad_frames` / `video_ok`: frame con NaN sau noi suy -> video khong ok neu co
  bat ky bad frame nao.

## Depth (tuy chon, `depth.enabled: true`)

- Reader: PKU `<video>-depth.avi` / TSU `<video>-depth.mp4` (dat `paths.depth_dir`).
- PKU depth avi la gray 8-bit (byte cao cua mm): Z(m) = gray * `depth.gray_to_m`
  (0.256). KHONG hieu chinh bang GT trong pipeline (doc lap de so sanh khach quan);
  `DepthProcessor.fit_with_gt()` la cong cu danh gia rieng.
- Ghost Legs Masking (auto): pixel trong bbox person gan hon dang ke so voi nguoi ->
  to xam truoc khi BlazePose (`depth.masking.*`).
- Back-projection 3D goc camera: `output.coordinate_mode: "camera"` +
  `depth.intrinsics`; Z loi -> bu tu khop cha. TSU (depth mp4 8-bit lossy, khong
  met that): kiem chung tinh tuyen tinh bang `scripts/check_tsu_depth.py` truoc khi
  quyet dinh; mac dinh TSU nen giu `world`.

## Visualize kiem chung — THEO CHUNK (`scripts/visualize_bad_frames.py`)

```powershell
.venv\Scripts\python scripts\visualize_bad_frames.py --config config.local.yaml --video 0016-R
```

- Voi moi chunk co frame khong ok: `failed_frames/<video>/chunk_NNN/` chua
  `<video>_chunk_NNN.mp4` (video cat vung chunk, ve bbox+conf+skeleton+joint conf
  len moi frame) + `frame_XXXXXX.jpg` (anh tung frame khong ok).
- Quy tac ve: khong bbox -> nguyen; bbox conf < threshold -> ve bbox; joint conf
  < threshold -> ve joint do; joint ok ve xanh.

## Cau truc

```
config.yaml                 # Toan bo tham so (cam hardcode trong code)
src/                        # Module OOP: orchestrator, pose_extractor, yolo_detector,
                            # tracker (DeepOCSORT/IoU), mapper, validator, interpolator,
                            # scaler, depth_processor, saver, logger, progress, ...
scripts/run_extraction.py   # CLI chay pipeline
scripts/split_actions.py    # Tach chunk -> file theo action (theo segments goc)
scripts/merge_actions.py    # Gop chunk cua 1 video
scripts/visualize_bad_frames.py  # Kiem chung theo chunk
scripts/check_tsu_depth.py  # Kiem chung tuyen tinh depth TSU
notebooks/                  # Notebook Colab (extract + demo overlay)
tests/test_pipeline.py      # Unit tests (pytest)
```

## Chay local (Windows, CPU)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest tests/ -v
.venv\Scripts\python scripts\run_extraction.py --config config.yaml --dataset PKU --start 0 --end 2 --max-segments 3
```

## Output

```
out/skeletons/
  chunks/npy/    <video>_chunk_<NNN>.npy     # skeleton theo chunk (union frames)
  chunks/jsonl/  <video>_chunk_<NNN>.jsonl   # metadata tung frame cua chunk
  chunks/meta/   <video>_chunk_<NNN>.json    # segments goc cua chunk (cho split)
  actions/npy/   <video>_NNN.npy             # skeleton theo TUNG ACTION (split_actions)
  actions/jsonl/ <video>_NNN.jsonl
  progress.json                              # resume
```

- Skeleton `.npy`: float32, shape `(T, 2, 25, 3)` — 25 khop NTU (giu nguyen
  SpineShoulder). Toa do tuy `output.coordinate_mode`, da duoc **chuan hoa ty le**
  (chia cho `mean |SpineMid - SpineBase|` theo tung person — bat bien voi khoang
  cach camera; tat qua `scaling.enabled: false`). Khop loi = NaN (chuoi loi >
  `interpolation.max_gap` frame giu NaN; frame TN = 0). Viec bo joint/resample 192
  cho SkateFormer do project `dataset-processing` dam nhiem.
- `.jsonl`: moi dong 1 frame: `frame_id`, `timestamp`, `action_label` (int, hoac
  list[int] tai vung overlap), `error_flags`, `landmarks_2d` (normalized),
  `bboxes` (xyxy+conf), `joint_conf` (visibility 25 khop tung person), `ok`.
- `failed_frames/<video>/`: anh loi 3 loai + `chunk_NNN/` (viz kiem chung).
- `logs/`: `<video>.log` + `batch_summary_<start>_<end>.csv`.

## Tach action tu chunk

```powershell
.venv\Scripts\python scripts\split_actions.py --output-dir out/skeletons --video 0016-R
```
