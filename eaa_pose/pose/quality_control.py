"""
eaa_pose.pose.quality_control
==============================
Skeleton quality control (QC) pipeline.

Applied to a full-video skeleton sequence *before* slicing into per-sample
segments.  The pipeline runs in-place on a ``(T, M, 25, 6)`` array where
the last axis is ``[x, y, z, confidence, valid_mask, reconstructed_flag]``.

Pipeline stages
---------------
1. **Confidence threshold** — joints with ``confidence < min_conf`` are
   flagged as invalid (``valid_mask = 0``).
2. **Missing-ratio check** — frames where the fraction of invalid joints
   exceeds ``max_missing_ratio`` are flagged at the frame level.
3. **Short-gap interpolation** — linear interpolation over contiguous
   missing-joint runs of length ≤ ``max_interp_gap`` frames;
   interpolated joints get ``reconstructed_flag = 1``.
4. **Temporal smoothing** — Savitzky-Golay filter applied to each
   coordinate channel independently (only on valid + reconstructed joints).
5. **Normalization** — root-center (SpineBase = joint 0) subtraction +
   scale normalization by torso length (SpineBase ↔ SpineShoulder distance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from scipy.signal import savgol_filter  # type: ignore
except Exception as exc:  # noqa: BLE001
    savgol_filter = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

if TYPE_CHECKING:
    pass


# Channel indices in the 6-channel array
_X    = 0
_Y    = 1
_Z    = 2
_CONF = 3
_MASK = 4
_REC  = 5

# NTU-120 joint indices
_SPINE_BASE       = 0
_SPINE_SHOULDER   = 20

_NTU25_NAMES = [
    "SpineBase",
    "SpineMid",
    "Neck",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "LHand",
    "RShoulder",
    "RElbow",
    "RWrist",
    "RHand",
    "LHip",
    "LKnee",
    "LAnkle",
    "LFoot",
    "RHip",
    "RKnee",
    "RAnkle",
    "RFoot",
    "SpineShoulder",
    "LHandTip",
    "LThumb",
    "RHandTip",
    "RThumb",
]


class SkeletonQC:
    """Quality-control and normalization pipeline for 3-D skeleton sequences.

    Parameters
    ----------
    min_conf:
        Keypoints with confidence below this are marked invalid.
    max_missing_ratio:
        Maximum fraction of invalid joints per frame before the frame
        is flagged (does not remove the frame — the model decides).
    max_interp_gap:
        Maximum number of consecutive missing frames to interpolate.
    smooth_window:
        Savitzky-Golay filter window length (must be odd, ≥ 3).
        Set to 0 to disable smoothing.
    smooth_polyorder:
        Polynomial order for the Savitzky-Golay filter.
    normalize:
        If True, apply root-centering and torso-length normalization.
    """

    def __init__(
        self,
        min_conf:          float = 0.3,
        max_missing_ratio: float = 0.4,
        max_interp_gap:    int   = 10,
        smooth_window:     int   = 5,
        smooth_polyorder:  int   = 2,
        normalize:         bool  = True,
    ) -> None:
        self.min_conf          = min_conf
        self.max_missing_ratio = max_missing_ratio
        self.max_interp_gap    = max_interp_gap
        self.smooth_window     = smooth_window
        self.smooth_polyorder  = smooth_polyorder
        self.normalize         = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, sequence: np.ndarray) -> np.ndarray:
        """Apply the full QC pipeline to a skeleton sequence.

        Parameters
        ----------
        sequence:
            Array of shape ``(T, M, 25, 6)`` — raw skeleton sequence
            where channel 3 is confidence and channels 4–5 are
            initialised to 1 / 0 respectively.

        Returns
        -------
        Processed array of the same shape with ``valid_mask`` and
        ``reconstructed_flag`` channels filled in.
        """
        seq = sequence.copy()

        self._apply_confidence_threshold(seq)
        self._interpolate_gaps(seq)
        self._temporal_smooth(seq)
        if self.normalize:
            self._normalize(seq)

        return seq

    def summarize(
        self,
        sequence: np.ndarray,
        frame_statuses: list[str] | None = None,
        long_gap_threshold: int | None = None,
        include_frame_statuses: bool = False,
    ) -> dict:
        """Return review-oriented quality statistics for a QC'd sequence.

        The summary is intentionally non-destructive: it never marks a
        sample for exclusion.  ``needs_review`` only means that the sample
        contains a long invalid region or low-validity pattern worth checking.
        """
        seq = sequence
        T, M, J, _ = seq.shape
        threshold = self.max_interp_gap if long_gap_threshold is None else long_gap_threshold

        mask = seq[..., _MASK] > 0.5
        rec = seq[..., _REC] > 0.5

        total_points = max(1, T * M * J)
        valid_ratio = float(mask.sum() / total_points)
        reconstructed_ratio = float(rec.sum() / total_points)

        frame_valid_ratio = mask.mean(axis=2)  # (T, M)
        full_body_missing = frame_valid_ratio == 0.0
        high_missing = (1.0 - frame_valid_ratio) > self.max_missing_ratio

        longest_full_body_gap = 0
        longest_high_missing_gap = 0
        for m in range(M):
            longest_full_body_gap = max(
                longest_full_body_gap,
                self._longest_true_run(full_body_missing[:, m]),
            )
            longest_high_missing_gap = max(
                longest_high_missing_gap,
                self._longest_true_run(high_missing[:, m]),
            )

        longest_joint_gap = 0
        joint_gaps: list[dict] = []
        for m in range(M):
            for j in range(J):
                invalid = ~mask[:, m, j]
                gap = self._longest_true_run(invalid)
                longest_joint_gap = max(longest_joint_gap, gap)
                if gap > threshold:
                    joint_gaps.append(
                        {
                            "person_index": int(m),
                            "joint_index": int(j),
                            "joint_name": _NTU25_NAMES[j] if j < len(_NTU25_NAMES) else str(j),
                            "longest_invalid_gap": int(gap),
                            "valid_ratio": float(mask[:, m, j].mean()),
                        }
                    )

        joint_gaps.sort(key=lambda item: item["longest_invalid_gap"], reverse=True)

        frame_status_counts: dict[str, int] = {}
        if frame_statuses is not None:
            for status in frame_statuses:
                frame_status_counts[status] = frame_status_counts.get(status, 0) + 1

        needs_review = (
            longest_joint_gap > threshold
            or longest_full_body_gap > threshold
            or longest_high_missing_gap > threshold
        )

        summary = {
            "num_frames": int(T),
            "num_persons": int(M),
            "num_joints": int(J),
            "valid_ratio": valid_ratio,
            "reconstructed_ratio": reconstructed_ratio,
            "full_body_missing_ratio": float(full_body_missing.sum() / max(1, T * M)),
            "high_missing_frame_ratio": float(high_missing.sum() / max(1, T * M)),
            "long_gap_threshold": int(threshold),
            "longest_joint_invalid_gap": int(longest_joint_gap),
            "longest_full_body_invalid_gap": int(longest_full_body_gap),
            "longest_high_missing_gap": int(longest_high_missing_gap),
            "joints_with_long_invalid_gap": joint_gaps[:25],
            "frame_status_counts": frame_status_counts,
            "needs_review": bool(needs_review),
        }

        if include_frame_statuses and frame_statuses is not None:
            summary["frame_statuses"] = list(frame_statuses)

        return summary

    # ------------------------------------------------------------------
    # Pipeline stages (operate in-place on seq)
    # ------------------------------------------------------------------

    def _apply_confidence_threshold(self, seq: np.ndarray) -> None:
        """Mark low-confidence joints as invalid (valid_mask = 0)."""
        # seq shape: (T, M, 25, 6)
        low_conf = seq[..., _CONF] < self.min_conf
        seq[low_conf, _MASK] = 0.0
        seq[low_conf, _X:_Z+1] = 0.0   # zero out coordinates too

    def _interpolate_gaps(self, seq: np.ndarray) -> None:
        """Linear interpolation over short missing-joint runs.

        For each person ``m`` and joint ``j``, if there is a run of
        ``valid_mask == 0`` frames of length ≤ ``max_interp_gap`` AND
        the run is bounded by valid frames on both sides, fill the gap
        with linear interpolation and set ``reconstructed_flag = 1``.
        """
        T, M, J, _ = seq.shape

        for m in range(M):
            for j in range(J):
                valid = seq[:, m, j, _MASK].astype(bool)  # (T,)
                frames = np.arange(T)

                # Find contiguous invalid runs
                i = 0
                while i < T:
                    if valid[i]:
                        i += 1
                        continue
                    # Start of invalid run
                    run_start = i
                    while i < T and not valid[i]:
                        i += 1
                    run_end = i - 1   # inclusive

                    gap_len = run_end - run_start + 1
                    if gap_len > self.max_interp_gap:
                        continue   # too long to interpolate

                    # Check bounding valid frames
                    left  = run_start - 1
                    right = run_end   + 1
                    if left < 0 or right >= T:
                        continue   # at boundary — cannot interpolate

                    if not valid[left] or not valid[right]:
                        continue

                    # Interpolate coordinates x, y, z linearly
                    for ch in (_X, _Y, _Z):
                        left_val  = seq[left,  m, j, ch]
                        right_val = seq[right, m, j, ch]
                        for k, t in enumerate(range(run_start, run_end + 1)):
                            alpha = (k + 1) / (gap_len + 1)
                            seq[t, m, j, ch] = (1 - alpha) * left_val + alpha * right_val

                    # Interpolate confidence
                    left_conf  = seq[left,  m, j, _CONF]
                    right_conf = seq[right, m, j, _CONF]
                    for k, t in enumerate(range(run_start, run_end + 1)):
                        alpha = (k + 1) / (gap_len + 1)
                        seq[t, m, j, _CONF] = (1 - alpha) * left_conf + alpha * right_conf
                        seq[t, m, j, _MASK] = 1.0
                        seq[t, m, j, _REC]  = 1.0   # mark as reconstructed

    def _temporal_smooth(self, seq: np.ndarray) -> None:
        """Apply Savitzky-Golay smoothing along the time axis.

        Only valid (mask=1) frames contribute to the smooth; the filter
        is applied per-joint per-channel independently.
        """
        if self.smooth_window < 3:
            return
        if savgol_filter is None:
            warnings.warn(
                f"Skipping temporal smoothing because scipy.signal.savgol_filter "
                f"could not be imported: {_SCIPY_IMPORT_ERROR}",
                stacklevel=2,
            )
            return

        T, M, J, _ = seq.shape
        win = self.smooth_window
        if T < win:
            return   # sequence too short to smooth

        for m in range(M):
            for j in range(J):
                for ch in (_X, _Y, _Z):
                    signal = seq[:, m, j, ch]
                    mask   = seq[:, m, j, _MASK].astype(bool)

                    if mask.sum() < win:
                        continue   # not enough valid frames

                    # Smooth only valid + reconstructed frames
                    smoothed = savgol_filter(
                        signal,
                        window_length=win,
                        polyorder=self.smooth_polyorder,
                        mode="nearest",
                    )
                    seq[:, m, j, ch] = smoothed

    def _normalize(self, seq: np.ndarray) -> None:
        """Root-center + torso-length normalization.

        Steps per frame:
        1. Subtract SpineBase (joint 0) from all joint coordinates.
        2. Compute torso length = distance(SpineBase, SpineShoulder)
           averaged over valid frames.
        3. Divide all coordinates by torso length (clamped to avoid
           division by near-zero).

        Normalization is applied per-person independently.
        """
        T, M, J, _ = seq.shape

        for m in range(M):
            # Compute mean torso length across valid frames
            spine_base     = seq[:, m, _SPINE_BASE,     :3]  # (T, 3)
            spine_shoulder = seq[:, m, _SPINE_SHOULDER, :3]  # (T, 3)

            spine_base_valid     = seq[:, m, _SPINE_BASE,     _MASK].astype(bool)
            spine_shoulder_valid = seq[:, m, _SPINE_SHOULDER, _MASK].astype(bool)
            both_valid = spine_base_valid & spine_shoulder_valid

            if both_valid.sum() == 0:
                continue   # no valid reference frames for this person

            torso_vecs = spine_shoulder[both_valid] - spine_base[both_valid]
            torso_len = float(np.linalg.norm(torso_vecs, axis=-1).mean())
            torso_len = max(torso_len, 1e-3)   # avoid division by zero

            # Root-centering: subtract SpineBase from all joints
            root = spine_base[:, np.newaxis, :]  # (T, 1, 3)
            seq[:, m, :, :3] -= root

            # Scale normalization
            seq[:, m, :, :3] /= torso_len

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> int:
        """Return the longest contiguous run of True values."""
        longest = 0
        current = 0
        for value in values.astype(bool):
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)
