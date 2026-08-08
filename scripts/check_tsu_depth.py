"""Kiem chung tinh tuyen tinh cua depth TSU (gray 8-bit) ma KHONG can GT.

Y tuong: voi cung 1 nguoi o cac khoang cach khac nhau, chieu rong bbox ti le
nghich voi khoang cach that (width ~ 1/Z). Neu gray ti le tuyen tinh voi Z
(gray ~ a*Z + b) thi width * gray ~ const -> corr(width, gray) gan -1 (hoac
+1 neu gray nghich voi Z) va CV(width*gray) nho.

Vi du:
    python scripts/check_tsu_depth.py --config config.local-tsu.yaml --video P02T01C06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_manager import ConfigManager
from src.yolo_detector import YOLODetector


def check_linearity(cfg: ConfigManager, video_name: str, sample_step: int = 50) -> dict:
    video_dir = Path(cfg.get("paths.video_dir"))
    depth_dir = Path(cfg.get("paths.depth_dir"))
    rgb = cv2.VideoCapture(str(video_dir / f"{video_name}.mp4"))
    dep = cv2.VideoCapture(str(depth_dir / f"{video_name}-depth.mp4"))
    if not rgb.isOpened() or not dep.isOpened():
        raise IOError("Khong mo duoc RGB/depth video")
    yolo = YOLODetector(cfg)
    widths, grays = [], []
    n = int(rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    for fid in range(0, n, sample_step):
        rgb.set(cv2.CAP_PROP_POS_FRAMES, fid)
        dep.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok_r, frame = rgb.read()
        ok_d, dframe = dep.read()
        if not (ok_r and ok_d):
            continue
        boxes = yolo.detect(frame)
        if len(boxes) != 1:  # chi lay frame dung 1 nguoi
            continue
        b = boxes[0]
        x1, y1, x2, y2 = b.xyxy
        dgray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
        roi = dgray[int(y1 / 2):int(y2 / 2) or 1, int(x1 / 2):int(x2 / 2) or 1]
        valid = roi[roi > 0]
        if valid.size == 0:
            continue
        widths.append(x2 - x1)
        grays.append(float(np.median(valid)))
    rgb.release()
    dep.release()
    if len(widths) < 5:
        return {"n_samples": len(widths), "conclusion": "khong du mau"}
    w = np.array(widths, dtype=np.float64)
    g = np.array(grays, dtype=np.float64)
    corr = float(np.corrcoef(w, g)[0, 1])
    prod = w * g
    cv = float(prod.std() / prod.mean()) if prod.mean() > 0 else float("inf")
    linear = abs(corr) > 0.7 and cv < 0.35
    return {
        "n_samples": len(w),
        "corr(width, gray)": round(corr, 3),
        "CV(width*gray)": round(cv, 3),
        "gray_min_max": [float(g.min()), float(g.max())],
        "linear_enough_for_backprojection": bool(linear),
        "conclusion": "gray TUYEN TINH voi Z -> TSU back-project duoc (don vi tuong doi)"
        if linear else "gray KHONG tuyen tinh ro voi Z -> TSU nen giu BlazePose world",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiem chung tuyen tinh depth TSU")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-step", type=int, default=50)
    args = parser.parse_args()
    cfg = ConfigManager(args.config)
    import json
    print(json.dumps(check_linearity(cfg, args.video, args.sample_step),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
