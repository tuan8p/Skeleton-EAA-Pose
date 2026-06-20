"""
eaa_pose.pose.estimator_rtmw3d
================================
RTMW3D whole-body pose estimator wrapper.

RTMW3D (Real-Time Multi-person Whole-body 3D) estimates 133 3-D keypoints
per person from a cropped bounding box.  This module wraps the MMPose API
and returns keypoints in a uniform numpy array format.

Keypoint convention
-------------------
MMPose RTMW3D outputs keypoints in the COCO-Wholebody order (133 points):

    0-16   : body (17 pts, COCO)
    17-22  : foot (6 pts)
    23-90  : face (68 pts)
    91-112 : left hand (21 pts)
    113-132: right hand (21 pts)

The downstream :class:`~eaa_pose.pose.mapping_25.NTU25Mapper` converts
these 133 points to the NTU-120 / PKU-MMD 25-joint format.

Dependencies
------------
- mmpose >= 1.3  (install via ``mim install mmpose``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PoseResult:
    """Pose estimation result for a single person in one frame.

    Attributes
    ----------
    track_id:
        Tracker identity (from :class:`~eaa_pose.pose.tracker.TrackedPerson`).
    keypoints:
        Numpy array of shape ``(133, 4)`` — columns are
        ``[x, y, z, confidence]``.  Coordinates are in metres for the
        z-axis (metric depth from camera) and pixel-space for x, y
        (before normalisation).
    bbox:
        Source bounding box ``[x1, y1, x2, y2]``.
    """

    track_id: int
    keypoints: np.ndarray   # (133, 4)  float32
    bbox: np.ndarray        # (4,)      float32


class RTMW3DEstimator:
    """Runs RTMW3D pose estimation on person bounding boxes.

    Parameters
    ----------
    config:
        Path to the MMPose model config file.
    checkpoint:
        Path to (or URL of) the model checkpoint.
    device:
        Torch device string (e.g. ``'cuda'``, ``'cpu'``).
    keypoint_conf_threshold:
        Minimum keypoint confidence to treat a keypoint as valid.
        Keypoints below this threshold get ``confidence=0`` in the output.

    Notes
    -----
    Model loading is deferred to the first :meth:`estimate` call so that
    the class can be instantiated on CPU-only machines without failing.
    """

    # Number of keypoints in the COCO-Wholebody topology
    N_KEYPOINTS = 133

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        keypoint_conf_threshold: float = 0.3,
    ) -> None:
        self._config     = str(config)
        self._checkpoint = str(checkpoint)
        self._device     = device
        self._kp_thresh  = keypoint_conf_threshold
        self._model      = None   # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        tracked_persons: list,   # list[TrackedPerson]
    ) -> list[PoseResult]:
        """Estimate 3-D whole-body pose for all tracked persons in a frame.

        Parameters
        ----------
        frame:
            BGR uint8 image of shape ``(H, W, 3)``.
        tracked_persons:
            List of :class:`~eaa_pose.pose.tracker.TrackedPerson` from the
            tracker.  Each person's bbox is used as the crop region.

        Returns
        -------
        List of :class:`PoseResult`, one per person, in the same order
        as ``tracked_persons``.
        """
        if self._model is None:
            self._load_model()

        if not tracked_persons:
            return []

        from mmpose.apis import inference_topdown  # type: ignore
        from mmpose.structures import merge_data_samples  # type: ignore

        # Build input list for mmpose (list of dicts with 'bbox')
        bboxes_xyxy = np.stack(
            [p.bbox for p in tracked_persons], axis=0
        )  # (M, 4)

        results = inference_topdown(self._model, frame, bboxes_xyxy)
        merged  = merge_data_samples(results)

        return self._parse_results(merged, tracked_persons)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the MMPose RTMW3D model."""
        try:
            from mmpose.apis import init_model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mmpose is not installed. Install with: mim install mmpose"
            ) from exc

        self._model = init_model(
            self._config,
            self._checkpoint,
            device=self._device,
        )

    def _parse_results(
        self,
        merged_result,
        tracked_persons: list,
    ) -> list[PoseResult]:
        """Convert MMPose output to a list of :class:`PoseResult`.

        Expected MMPose v1 output structure::

            merged_result.pred_instances.keypoints        (M, 133, 2)  x,y
            merged_result.pred_instances.keypoints_3d     (M, 133, 3)  x,y,z
            merged_result.pred_instances.keypoint_scores  (M, 133)

        If ``keypoints_3d`` is not available (2-D model), ``z`` is set to 0.
        """
        pred = merged_result.pred_instances

        kps_2d  = pred.keypoints.cpu().numpy()        # (M, 133, 2)
        scores  = pred.keypoint_scores.cpu().numpy()  # (M, 133)

        has_3d = hasattr(pred, "keypoints_3d") and pred.keypoints_3d is not None
        if has_3d:
            kps_3d = pred.keypoints_3d.cpu().numpy()  # (M, 133, 3)
        else:
            kps_3d = np.zeros((*kps_2d.shape[:2], 3), dtype=np.float32)
            kps_3d[..., :2] = kps_2d

        pose_results: list[PoseResult] = []
        M = min(kps_2d.shape[0], len(tracked_persons))

        for i in range(M):
            # Build (133, 4) array: [x, y, z, confidence]
            kps = np.zeros((self.N_KEYPOINTS, 4), dtype=np.float32)
            kps[:, 0] = kps_3d[i, :, 0]   # x
            kps[:, 1] = kps_3d[i, :, 1]   # y
            kps[:, 2] = kps_3d[i, :, 2]   # z
            kps[:, 3] = scores[i]          # confidence

            # Zero out low-confidence keypoints
            low_conf_mask = kps[:, 3] < self._kp_thresh
            kps[low_conf_mask, 3] = 0.0

            pose_results.append(
                PoseResult(
                    track_id=tracked_persons[i].track_id,
                    keypoints=kps,
                    bbox=tracked_persons[i].bbox,
                )
            )

        return pose_results
