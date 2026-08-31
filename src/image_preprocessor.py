"""Image preprocessing according to config: resize (letterbox), CLAHE, gamma, denoise.

process() tra ve (frame_da_xu_ly, transform) voi transform = (scale, pad_x, pad_y)
de anh xa toa do 2D normalized tu anh da resize ve frame goc:
    x_goc = (x_resized * W_out - pad_x) / (scale * W_goc)
"""
from __future__ import annotations

import cv2
import numpy as np

from .config_manager import ConfigManager


class ImagePreprocessor:
    def __init__(self, cfg: ConfigManager):
        self.resize_on = bool(cfg.get("preprocessing.resize.enabled", False))
        self.out_w = int(cfg.get("preprocessing.resize.width", 640))
        self.out_h = int(cfg.get("preprocessing.resize.height", 640))
        self.keep_ratio = bool(cfg.get("preprocessing.resize.keep_ratio", True))
        self.clahe_on = bool(cfg.get("preprocessing.clahe.enabled", False))
        self.clahe_clip = float(cfg.get("preprocessing.clahe.clip_limit", 2.0))
        self.clahe_grid = int(cfg.get("preprocessing.clahe.tile_grid_size", 8))
        self.gamma_on = bool(cfg.get("preprocessing.gamma.enabled", False))
        self.gamma_val = float(cfg.get("preprocessing.gamma.value", 0.8))
        self.denoise_on = bool(cfg.get("preprocessing.denoise.enabled", False))
        self.denoise_method = str(cfg.get("preprocessing.denoise.method", "bilateral"))
        self.denoise_strength = int(cfg.get("preprocessing.denoise.strength", 5))

    def process_retry1(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
        out = frame.copy()
        if self.gamma_on:
            table = (np.arange(256, dtype=np.float32) / 255.0) ** self.gamma_val * 255.0
            out = cv2.LUT(out, table.astype(np.uint8))
        if self.clahe_on:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                    tileGridSize=(self.clahe_grid, self.clahe_grid))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        h0, w0 = out.shape[:2]
        return out, (1.0, 0, 0, w0, h0)

    def process_retry2(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
        out = frame.copy()
        if self.denoise_on:
            k = self.denoise_strength | 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        
        h0, w0 = out.shape[:2]
        size = max(h0, w0)
        canvas = np.zeros((size, size, 3), dtype=out.dtype)
        pad_x, pad_y = (size - w0) // 2, (size - h0) // 2
        canvas[pad_y:pad_y + h0, pad_x:pad_x + w0] = out
        return canvas, (1.0, pad_x, pad_y, size, size)

    @staticmethod
    def to_original_coords(norm_xy: np.ndarray, orig_wh: tuple[int, int],
                           transform: tuple[float, int, int, int, int]) -> np.ndarray:
        """Map normalized coordinates from preprocessed image back to original frame coordinates."""
        scale, pad_x, pad_y, out_w, out_h = transform
        w0, h0 = orig_wh
        xy = norm_xy.copy()
        xy[:, 0] = (norm_xy[:, 0] * out_w - pad_x) / (scale * w0)
        xy[:, 1] = (norm_xy[:, 1] * out_h - pad_y) / (scale * h0)
        return xy
