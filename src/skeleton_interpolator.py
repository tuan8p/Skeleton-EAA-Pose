"""Noi suy temporal + spatial cho frame thieu khop.

Quy tac (PLAN muc 2 buoc 6): chuoi loi > max_gap frame lien tiep thi KHONG noi
suy (giu NaN). Temporal (tuyen tinh theo thoi gian) truoc, spatial sau cho cac
khop con NaN trong frame khong thuoc chuoi dai: uoc luong = khop cha + median
offset hoc tu cac frame hop le.
"""
from __future__ import annotations

import numpy as np

from .config_manager import ConfigManager
from .mapper import NUM_NTU_JOINTS

# khop cha dung cho noi suy spatial (theo cay xuong NTU)
SPATIAL_PARENT: dict[int, int] = {
    1: 0, 2: 20, 3: 2, 20: 1,
    4: 20, 5: 4, 6: 5, 7: 6, 21: 7, 22: 6,
    8: 20, 9: 8, 10: 9, 11: 10, 23: 11, 24: 10,
    12: 0, 13: 12, 14: 13, 15: 14,
    16: 0, 17: 16, 18: 17, 19: 18,
}


class SkeletonInterpolator:
    def __init__(self, cfg: ConfigManager):
        self.max_gap = int(cfg.get("temporal.empty_run_frames", 30))

    def interpolate(self, skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """skel (T, P, J, 3) -> (interpolated copy, filled[T]) where filled[t]=True
        if frame t had at least 1 NaN value filled."""
        before = np.isnan(skel)
        out = skel.copy()
        self._temporal(out)
        self._spatial(out)
        filled = (before & ~np.isnan(out)).reshape(out.shape[0], -1).any(axis=1)
        return out, filled

    def _temporal(self, skel: np.ndarray) -> None:
        T = skel.shape[0]
        flat = skel.reshape(T, -1)
        for col in range(flat.shape[1]):
            series = flat[:, col]
            isnan = np.isnan(series)
            if not isnan.any():
                continue
            idx = np.arange(T)
            valid = idx[~isnan]
            if len(valid) == 0:
                continue
            # tim tung chuoi NaN lien tiep
            starts = np.where(isnan & ~np.roll(isnan, 1))[0]
            ends = np.where(isnan & ~np.roll(isnan, -1))[0]
            for s, e in zip(starts, ends):
                gap = e - s + 1
                if gap > self.max_gap:
                    continue
                prev_ok = s - 1 in valid
                next_ok = e + 1 in valid
                if prev_ok and next_ok:
                    series[s:e + 1] = np.interp(idx[s:e + 1], [s - 1, e + 1],
                                                [series[s - 1], series[e + 1]])
                elif prev_ok:
                    series[s:e + 1] = series[s - 1]
                elif next_ok:
                    series[s:e + 1] = series[e + 1]

    def _spatial(self, skel: np.ndarray) -> None:
        T, P, J, C = skel.shape
        if J != NUM_NTU_JOINTS:
            return
        for p in range(P):
            person = skel[:, p]
            long_gap = self._long_gap_frames(person)
            for child, parent in SPATIAL_PARENT.items():
                missing = np.isnan(person[:, child, 0]) & ~long_gap
                if not missing.any():
                    continue
                ok = ~np.isnan(person[:, child, 0]) & ~np.isnan(person[:, parent, 0])
                if ok.sum() < 2:
                    continue
                offset = np.median(person[ok, child] - person[ok, parent], axis=0)
                fillable = missing & ~np.isnan(person[:, parent, 0])
                person[fillable, child] = person[fillable, parent] + offset

    def _long_gap_frames(self, person: np.ndarray) -> np.ndarray:
        """Danh dau frame thuoc chuoi > max_gap frame loi lien tiep (person-level)."""
        T = person.shape[0]
        frame_bad = np.isnan(person).reshape(T, -1).all(axis=1)
        long_gap = np.zeros(T, dtype=bool)
        idx = np.arange(T)
        starts = np.where(frame_bad & ~np.roll(frame_bad, 1))[0]
        ends = np.where(frame_bad & ~np.roll(frame_bad, -1))[0]
        for s, e in zip(starts, ends):
            if e - s + 1 > self.max_gap:
                long_gap[s:e + 1] = True
        return long_gap
