"""
eaa_pose.pose.detector
======================
Person detector wrapper using RTMDet from MMDetection.

The detector returns bounding boxes for every person visible in a frame.
Downstream, boxes are passed to :class:`~eaa_pose.pose.tracker.PersonTracker`
for identity-consistent tracking.

Dependencies
------------
- mmdet >= 3.2  (install via ``mim install mmdet``)
- mmcv >= 2.1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    """A single person detection result.

    Attributes
    ----------
    bbox:
        Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
    score:
        Detection confidence in ``[0, 1]``.
    """

    bbox: np.ndarray    # shape (4,)  float32
    score: float


class PersonDetector:
    """Wraps an RTMDet (or any MMDet) person detector.

    Parameters
    ----------
    config:
        Path to the MMDetection model config file.
    checkpoint:
        Path to (or URL of) the model checkpoint weights.
    score_threshold:
        Minimum detection score to keep a box.
    device:
        Torch device string (e.g. ``'cuda'``, ``'cpu'``).

    Notes
    -----
    The model is loaded lazily on the first call to :meth:`detect` so
    that import-time failures do not block non-GPU code paths.
    """

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        score_threshold: float = 0.3,
        device: str = "cuda",
    ) -> None:
        self._config    = str(config)
        self._checkpoint = str(checkpoint)
        self._score_threshold = score_threshold
        self._device    = device
        self._model     = None   # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single BGR frame.

        Parameters
        ----------
        frame:
            BGR image as a uint8 numpy array of shape ``(H, W, 3)``.

        Returns
        -------
        List of :class:`Detection` objects sorted by score descending.
        """
        if self._model is None:
            self._load_model()

        # mmdet inference_detector returns an InferenceResult object
        from mmdet.apis import inference_detector  # type: ignore

        result = inference_detector(self._model, frame)
        return self._parse_result(result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the MMDetection model (called on first use)."""
        try:
            from mmdet.apis import init_detector  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mmdet is not installed. Install with: mim install mmdet"
            ) from exc

        self._model = init_detector(
            self._config,
            self._checkpoint,
            device=self._device,
        )

    def _parse_result(self, result) -> list[Detection]:
        """Extract person bounding boxes from an MMDet inference result.

        MMDet v3 returns a ``DetDataSample`` object with
        ``pred_instances.bboxes`` and ``pred_instances.scores``.
        We keep only the 'person' class (COCO class 0).
        """
        detections: list[Detection] = []

        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()   # (N, 4)
        scores = pred.scores.cpu().numpy()   # (N,)
        labels = pred.labels.cpu().numpy()   # (N,)

        for bbox, score, label in zip(bboxes, scores, labels):
            if int(label) != 0:   # 0 = person in COCO
                continue
            if float(score) < self._score_threshold:
                continue
            detections.append(
                Detection(
                    bbox=bbox.astype(np.float32),
                    score=float(score),
                )
            )

        # Highest confidence first
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
