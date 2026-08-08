"""Chuan hoa ty le (scaling): dua toa do ve bat bien voi khoang cach camera.

Cong thuc theo dataset-processing (scale_by_spine), nhung GIU NGUYEN 25 khop
(khong bo SpineShoulder):
    scale_person = mean_t( |SpineMid_t - SpineBase_t| )   (hang so theo person)
    skel /= scale_person
Frame co NaN / person vang (toan 0) khong tham gia tinh mean va khong bi anh
huong (NaN/scale van NaN, 0/scale van 0).
"""
from __future__ import annotations

import numpy as np

from .config_manager import ConfigManager


class SkeletonScaler:
    def __init__(self, cfg: ConfigManager):
        self.enabled = bool(cfg.get("scaling.enabled", True))
        self.root_idx = int(cfg.get("scaling.root_idx", 0))    # SpineBase
        self.spine_idx = int(cfg.get("scaling.spine_idx", 1))  # SpineMid

    def scale(self, skel: np.ndarray) -> np.ndarray:
        """skel (T, P, J, 3) -> ban sao da scale theo tung person."""
        if not self.enabled:
            return skel
        out = skel.astype(np.float32, copy=True)
        for p in range(out.shape[1]):
            diff = out[:, p, self.spine_idx, :] - out[:, p, self.root_idx, :]
            distances = np.linalg.norm(diff, axis=1)
            valid = np.isfinite(distances) & (distances > 1e-6)
            scale = float(np.mean(distances[valid])) if valid.any() else 1.0
            if scale < 1e-6:
                scale = 1.0
            out[:, p] /= scale
        return out
