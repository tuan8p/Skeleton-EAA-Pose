"""Validate 25 khop da map: retry, danh dau NaN, phan loai loi frame."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config_manager import ConfigManager

ERR_NO_DETECT = "no_detect"
ERR_LOW_CONF = "low_conf"
ERR_POSE_FAIL = "pose_fail"


@dataclass
class FrameValidation:
    skeleton: np.ndarray          # (25, 3) - joint loi = NaN
    visibility: np.ndarray        # (25,)
    error_type: str | None        # None neu frame hop le
    nan_joints: list[int] = field(default_factory=list)


class Validator:
    def __init__(self, cfg: ConfigManager):
        self.conf_threshold = float(cfg.get("thresholds.confidence", 0.5))
        self.max_nan_ratio = float(cfg.get("thresholds.max_nan_ratio", 0.5))
        self.max_retries = int(cfg.get("validator.max_retries", 2))

    def needs_retry(self, visibility: np.ndarray) -> bool:
        """Bat ky joint nao co visibility < threshold -> can retry."""
        return bool(np.any(visibility < self.conf_threshold))

    def all_low_conf(self, visibility: np.ndarray) -> bool:
        return bool(np.all(visibility < self.conf_threshold))

    def finalize(self, skeleton: np.ndarray, visibility: np.ndarray) -> FrameValidation:
        """Sau khi het retry: joint visibility thap -> NaN; phan loai loi."""
        skel = skeleton.astype(np.float32, copy=True)
        bad = visibility < self.conf_threshold
        skel[bad] = np.nan
        nan_joints = [int(i) for i in np.where(bad)[0]]
        nan_ratio = len(nan_joints) / skeleton.shape[0]
        if nan_ratio > self.max_nan_ratio:
            return FrameValidation(skel, visibility, ERR_POSE_FAIL, nan_joints)
        return FrameValidation(skel, visibility, None, nan_joints)
