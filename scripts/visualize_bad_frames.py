"""Visualize kiem chung pipeline — THEO CHUNK (khong lan giua cac chunk).

Voi moi chunk co frame khong ok (ok=false):
  failed_frames/<video>/chunk_NNN/
    <video>_chunk_NNN.mp4   # video cat vung chunk [min_start..max_end], ve day du
                            # bbox + bbox conf + skeleton + joint conf len moi frame
    frame_XXXXXX.jpg        # anh tung frame khong ok cua chunk do

Quy tac ve anh frame fail:
- Khong detect duoc: khong co bbox -> anh nguyen; bbox conf < threshold -> ve bbox.
- Detect duoc ma pose khong duoc: khong co joint -> anh nguyen;
  joint conf < threshold -> ve cac joint yeu do (do), joint ok ve xanh.

Vi du:
    python scripts/visualize_bad_frames.py --config config.local.yaml --video 0016-R
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_manager import ConfigManager
from src.mapper import NTU_BONES

OK_COLOR = (0, 255, 0)
BAD_COLOR = (0, 0, 255)


def _draw_boxes(frame: np.ndarray, bboxes: list, conf_th: float) -> None:
    for b in bboxes or []:
        x1, y1, x2, y2, conf = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
        color = OK_COLOR if conf >= conf_th else BAD_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _draw_skeleton(frame: np.ndarray, lm2d, joint_conf, conf_th: float) -> None:
    """Ve skeleton; joint yeu (conf < threshold) ve mau do + ghi conf."""
    h, w = frame.shape[:2]
    persons_lm = lm2d if isinstance(lm2d[0][0], list) else [lm2d]
    persons_cf = joint_conf if joint_conf else [None] * len(persons_lm)
    for pts, confs in zip(persons_lm, persons_cf):
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        valid = ~np.isnan(pts).any(axis=1)
        px = np.full((len(pts), 2), -1, dtype=int)
        px[valid, 0] = (pts[valid, 0] * w).astype(int)
        px[valid, 1] = (pts[valid, 1] * h).astype(int)
        for a, b in NTU_BONES:
            if a < len(px) and b < len(px) and valid[a] and valid[b]:
                cv2.line(frame, tuple(px[a]), tuple(px[b]), OK_COLOR, 1)
        for j in range(len(px)):
            if not valid[j]:
                continue
            conf = confs[j] if confs is not None and j < len(confs) and \
                confs[j] is not None else None
            weak = conf is not None and conf < conf_th
            cv2.circle(frame, tuple(px[j]), 3, BAD_COLOR if weak else OK_COLOR, -1)
            if conf is not None:
                cv2.putText(frame, f"{conf:.2f}", (px[j, 0] + 3, px[j, 1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            BAD_COLOR if weak else OK_COLOR, 1)


def visualize_video(cfg: ConfigManager, video_name: str) -> dict:
    output_dir = Path(cfg.get("paths.output_dir"))
    video_dir = Path(cfg.get("paths.video_dir"))
    failed_dir = Path(cfg.get("paths.failed_frames_dir"))
    conf_th = float(cfg.get("thresholds.confidence", 0.3))
    video_ext = ".avi" if str(cfg.get("dataset", "PKU")).upper() == "PKU" else ".mp4"
    video_path = video_dir / f"{video_name}{video_ext}"

    chunk_meta_dir = output_dir / "chunks" / "meta"
    chunk_jsonl_dir = output_dir / "chunks" / "jsonl"
    report = {"video": video_name, "video_ok": True, "chunks": {}}

    for meta_path in sorted(chunk_meta_dir.glob(f"{video_name}_chunk_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ci = meta["chunk_index"]
        rows = [json.loads(l) for l in
                (chunk_jsonl_dir / f"{video_name}_chunk_{ci:03d}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = [r for r in rows if r.get("ok") is False]
        if not bad:
            continue  # fail-only
        report["video_ok"] = False
        seg_start = min(s["start"] for s in meta["segments"])
        seg_end = max(s["end"] for s in meta["segments"])
        rows_by_fid = {r["frame_id"]: r for r in rows}

        viz_dir = failed_dir / video_name / f"chunk_{ci:03d}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Khong mo duoc video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(viz_dir / f"{video_name}_chunk_{ci:03d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        bad_fids = set(r["frame_id"] for r in bad)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start - 1)
            for fid in range(seg_start, seg_end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                row = rows_by_fid.get(fid)
                if row is None:
                    writer.write(frame)
                    continue
                out = frame.copy()
                if row.get("bboxes"):
                    _draw_boxes(out, row["bboxes"], conf_th)
                if row.get("landmarks_2d") is not None:
                    _draw_skeleton(out, row["landmarks_2d"],
                                   row.get("joint_conf"), conf_th)
                if fid in bad_fids:
                    cv2.imwrite(str(viz_dir / f"frame_{fid:06d}.jpg"), out)
                writer.write(out)
        finally:
            cap.release()
            writer.release()
        report["chunks"][f"chunk_{ci:03d}"] = {
            "bad_frames": sorted(bad_fids),
            "viz_dir": str(viz_dir),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bad frames per chunk")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    cfg = ConfigManager(args.config)
    print(json.dumps(visualize_video(cfg, args.video), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
