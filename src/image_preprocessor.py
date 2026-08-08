"""Preprocessing anh theo config: resize (letterbox), CLAHE, gamma, denoise.

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

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
        h0, w0 = frame.shape[:2]
        scale, pad_x, pad_y = 1.0, 0, 0
        out = frame
        if self.resize_on:
            if self.keep_ratio:
                scale = min(self.out_w / w0, self.out_h / h0)
                new_w, new_h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
                resized = cv2.resize(out, (new_w, new_h))
                canvas = np.zeros((self.out_h, self.out_w, 3), dtype=resized.dtype)
                pad_x, pad_y = (self.out_w - new_w) // 2, (self.out_h - new_h) // 2
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                out = canvas
            else:
                out = cv2.resize(out, (self.out_w, self.out_h))
        if self.gamma_on:
            table = (np.arange(256, dtype=np.float32) / 255.0) ** self.gamma_val * 255.0
            out = cv2.LUT(out, table.astype(np.uint8))
        if self.clahe_on:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                    tileGridSize=(self.clahe_grid, self.clahe_grid))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if self.denoise_on:
            if self.denoise_method == "bilateral":
                out = cv2.bilateralFilter(out, self.denoise_strength,
                                          self.denoise_strength * 2, self.denoise_strength / 2)
            else:
                k = self.denoise_strength | 1
                out = cv2.GaussianBlur(out, (k, k), 0)
        return out, (scale, pad_x, pad_y)

    @staticmethod
    def to_original_coords(norm_xy: np.ndarray, orig_wh: tuple[int, int],
                           transform: tuple[float, int, int],
                           out_wh: tuple[int, int]) -> np.ndarray:
        """Anh xa toa do normalized tren anh da preprocess ve normalized tren frame goc."""
        scale, pad_x, pad_y = transform
        w0, h0 = orig_wh
        out_w, out_h = out_wh
        xy = norm_xy.copy()
        xy[:, 0] = (norm_xy[:, 0] * out_w - pad_x) / (scale * w0)
        xy[:, 1] = (norm_xy[:, 1] * out_h - pad_y) / (scale * h0)
        return xy
