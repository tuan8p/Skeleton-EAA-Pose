"""Xu ly Depth cho pipeline (PKU-MMD / TSU).

- Reader: PKU `<video>-infrared.avi` (512x424), TSU `<video>-depth.mp4` (320x240);
  ca hai deu la video gray 8-bit -> depth tuong doi (don vi thang xam).
- Align: resize depth ve kich thuoc RGB (TSU x2; PKU 512x424 -> 1920x1080).
- Ghost Legs Masking (auto): pixel trong bbox person co depth "gan hon dang ke"
  so voi depth tham chieu cua nguoi -> to mau xam truoc khi BlazePose.
- Back-projection: (x_px, y_px, Z) + intrinsics -> (X, Y, Z) goc camera;
  Z khong hop le (<=0) -> bu bang Z khop cha (SPATIAL_PARENT), cuoi cung -> NaN.
- Hieu chinh gray->met bang skeleton GT PKU (tuy chon, phuc vu kiem chung).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config_manager import ConfigManager
from .skeleton_interpolator import SPATIAL_PARENT

# Intrinsics depth camera Kinect v2 (xap xi cong khai), dung de chieu GT 3D -> pixel depth
_KINECT_V2_DEPTH_INTR = dict(fx=367.8, fy=366.8, cx=256.6, cy=209.3)


class DepthProcessor:
    def __init__(self, cfg: ConfigManager):
        self.enabled = bool(cfg.get("depth.enabled", False))
        self.depth_dir = Path(str(cfg.get("paths.depth_dir", "")))
        self.is_pku = str(cfg.get("dataset", "PKU")).upper() == "PKU"
        self.scale_to_rgb = bool(cfg.get("depth.scale_to_rgb", True))
        self.masking_on = bool(cfg.get("depth.masking.enabled", True))
        self.min_delta = float(cfg.get("depth.masking.min_delta", 15))
        self.fill_color = tuple(int(v) for v in cfg.get("depth.masking.fill_color",
                                                        [128, 128, 128]))
        self.intr = {k: float(cfg.get(f"depth.intrinsics.{k}", v)) for k, v in
                     dict(fx=1059.29, fy=1059.32, cx=962.90, cy=543.40).items()}
        self.gray_to_m = float(cfg.get("depth.gray_to_m", 0.256))
        self._cap: cv2.VideoCapture | None = None
        self._cur_fid = 0

    # ---------------- reader ----------------
    def open(self, video_name: str) -> bool:
        """Mo depth video tuong ung. Tra ve False neu khong co file."""
        if not self.enabled:
            return False
        if self.is_pku:
            candidates = [self.depth_dir / f"{video_name}-depth.avi",
                          self.depth_dir / f"{video_name}-infrared.avi",
                          self.depth_dir / f"{video_name}.avi"]
        else:
            candidates = [self.depth_dir / f"{video_name}-depth.mp4"]
        for p in candidates:
            if p.exists():
                self._cap = cv2.VideoCapture(str(p))
                self._cur_fid = 0
                return self._cap.isOpened()
        return False

    def read(self, fid: int, to_meters: bool = False) -> np.ndarray | None:
        """Doc frame depth (1-based, dong bo RGB) -> gray float32 (H, W).
        to_meters=True: nhan gray_to_m (chi PKU; TSU giu thang xam tuong doi)."""
        if self._cap is None:
            return None
        if fid != self._cur_fid + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._cur_fid = fid
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth = frame.astype(np.float32)
        if to_meters and self.is_pku:
            depth = depth * self.gray_to_m
        return depth

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def align(self, depth: np.ndarray, rgb_wh: tuple[int, int]) -> np.ndarray:
        """Resize depth ve kich thuoc RGB (align don gian cung FOV)."""
        if not self.scale_to_rgb:
            return depth
        w, h = rgb_wh
        if (depth.shape[1], depth.shape[0]) == (w, h):
            return depth
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---------------- ghost legs masking ----------------
    def mask_obstruction(self, frame: np.ndarray,
                         boxes: list[tuple[int, int, int, int]],
                         depth: np.ndarray) -> np.ndarray:
        """Auto: chi mask khi co vung "gan hon nguoi" dang ke trong bbox.

        depth_ref = median depth trong vung trung tam bbox (40% giua).
        Vat can = pixel co 0 < depth < depth_ref - min_delta chiem > 1% bbox.
        """
        if not self.masking_on or not boxes:
            return frame
        out = frame
        h, w = depth.shape[:2]
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = depth[y1:y2, x1:x2]
            ch, cw = roi.shape
            center = roi[int(ch * 0.3):int(ch * 0.7) or 1, int(cw * 0.3):int(cw * 0.7) or 1]
            valid = center[center > 0]
            if valid.size == 0:
                continue
            ref = float(np.median(valid))
            obstacle = (roi > 0) & (roi < ref - self.min_delta)
            if obstacle.mean() > 0.01:
                out = out.copy()
                out[y1:y2, x1:x2][obstacle] = self.fill_color
        return out

    # ---------------- back-projection ----------------
    def backproject(self, lm2d_px: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """(N,2) pixel frame goc + depth da align -> (N,3) goc camera.

        Z <= 0 (mat depth) -> bu bang Z khop cha (theo cay NTU); cuoi cung -> NaN.
        """
        h, w = depth.shape[:2]
        N = lm2d_px.shape[0]
        xs = np.clip(np.round(lm2d_px[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(lm2d_px[:, 1]).astype(int), 0, h - 1)
        z = depth[ys, xs]
        # bu Z tu khop cha khi Z loi (lap toi da 3 lan theo cay)
        for _ in range(3):
            bad = z <= 0
            if not bad.any():
                break
            for child in np.where(bad)[0]:
                parent = SPATIAL_PARENT.get(int(child))
                if parent is not None and z[parent] > 0:
                    z[child] = z[parent]
        fx, fy, cx, cy = self.intr["fx"], self.intr["fy"], self.intr["cx"], self.intr["cy"]
        with np.errstate(invalid="ignore", divide="ignore"):
            x3 = (lm2d_px[:, 0] - cx) * z / fx
            y3 = (lm2d_px[:, 1] - cy) * z / fy
        out = np.stack([x3, y3, z], axis=1).astype(np.float32)
        out[z <= 0] = np.nan
        return out

    # ---------------- hieu chinh bang GT (kiem chung) ----------------
    def fit_with_gt(self, gt_skeleton: np.ndarray, depth_dir_is_depth=True,
                    max_frames: int = 200) -> dict:
        """Do tuong quan giua gray pixel va Z that cua GT (camera-space, met).

        gt_skeleton: (F, 2, 25, 3) met. Tra ve dict {corr, a, b} voi Z ~= a*gray + b
        (fit tuyen tinh tren cac khop co gray > 0). Dung de kiem chung file depth.
        """
        if self._cap is None:
            raise RuntimeError("Chua mo depth video")
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        grays, zs = [], []
        intr = _KINECT_V2_DEPTH_INTR
        for fid in range(1, total + 1, step):
            depth = self.read(fid)
            if depth is None or fid > gt_skeleton.shape[0]:
                continue
            for p in range(2):
                for j in range(gt_skeleton.shape[2]):
                    X, Y, Z = gt_skeleton[fid - 1, p, j]
                    if Z <= 0:
                        continue
                    u = int(round(intr["fx"] * X / Z + intr["cx"]))
                    v = int(round(intr["fy"] * Y / Z + intr["cy"]))
                    if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                        g = depth[v, u]
                        if g > 0:
                            grays.append(g)
                            zs.append(Z)
        if len(grays) < 20:
            return {"corr": 0.0, "a": 1.0, "b": 0.0, "n": len(grays)}
        grays = np.array(grays)
        zs = np.array(zs)
        corr = float(np.corrcoef(grays, zs)[0, 1])
        a, b = np.polyfit(grays, zs, 1)
        return {"corr": corr, "a": float(a), "b": float(b), "n": len(grays)}
