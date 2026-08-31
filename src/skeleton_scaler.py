"""Skeleton scaling: normalize coordinates to be invariant to camera distance.

Formula follows dataset-processing (scale_by_spine) but KEEPS all 25 joints
(does not remove SpineShoulder):
    scale_person = mean_t( |SpineMid_t - SpineBase_t| )   (constant per person)
    skel /= scale_person
Frames with NaN / absent person (all-zero) do not participate in mean calculation and are not
affected (NaN/scale stays NaN, 0/scale stays 0).
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
        """skel (T, P, J, 3) -> scaled copy per person."""
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
