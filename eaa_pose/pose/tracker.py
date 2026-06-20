"""
eaa_pose.pose.tracker
======================
Person tracker wrapper using ByteTrack (via MMDetection's built-in tracker).

ByteTrack maintains consistent track IDs across frames by associating
detections with existing tracks using IoU and appearance cues.  We use
it here to keep a stable identity for each person so that skeleton
sequences are not accidentally mixed between subjects.

Dependencies
------------
- mmdet >= 3.2  (includes ByteTrack implementation)
- lap >= 0.4    (linear assignment solver used by ByteTrack)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedPerson:
    """A single tracked person in one frame.

    Attributes
    ----------
    track_id:
        Stable integer identity assigned by the tracker.
    bbox:
        Bounding box ``[x1, y1, x2, y2]`` in pixel coordinates.
    score:
        Association score (can be used to select top-M persons).
    """

    track_id: int
    bbox: np.ndarray   # shape (4,)  float32
    score: float


class PersonTracker:
    """Stateful ByteTrack-based person tracker.

    Wraps MMDetection's ``ByteTracker`` so that :class:`PersonDetector`
    output can be fed frame-by-frame to maintain consistent track IDs.

    Parameters
    ----------
    high_thresh:
        Minimum detection score to seed / keep a high-confidence track.
    low_thresh:
        Minimum score for a detection to be associated with an existing track.
    max_lost:
        Number of consecutive frames a track can be lost before removal.
    min_hits:
        Minimum confirmed frames before a new track is reported.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh:  float = 0.1,
        max_lost:    int   = 30,
        min_hits:    int   = 3,
    ) -> None:
        self._high_thresh = high_thresh
        self._low_thresh  = low_thresh
        self._max_lost    = max_lost
        self._min_hits    = min_hits
        self._tracker     = None   # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._tracker = None

    def update(
        self,
        detections: list,   # list[Detection] from PersonDetector
        frame_shape: tuple[int, int],   # (H, W)
    ) -> list[TrackedPerson]:
        """Update tracker with new detections and return active tracks.

        Parameters
        ----------
        detections:
            List of :class:`~eaa_pose.pose.detector.Detection` for the
            current frame.
        frame_shape:
            ``(height, width)`` of the current frame (needed by ByteTrack
            to normalise box coordinates).

        Returns
        -------
        List of :class:`TrackedPerson` — confirmed, active tracks for
        this frame, sorted by ``track_id``.
        """
        if self._tracker is None:
            self._load_tracker(frame_shape)

        if not detections:
            # Advance tracker with empty detections to age out lost tracks
            self._tracker.update([], frame_shape, frame_shape)
            return []

        bboxes = np.stack([d.bbox for d in detections], axis=0)     # (N, 4)
        scores = np.array([d.score for d in detections], dtype=np.float32)

        # ByteTracker.update expects bboxes in (x1,y1,x2,y2) format
        online_targets = self._tracker.update(
            bboxes,
            scores,
            frame_shape,
            frame_shape,
        )

        tracked: list[TrackedPerson] = []
        for t in online_targets:
            tracked.append(
                TrackedPerson(
                    track_id=int(t.track_id),
                    bbox=np.array(t.tlbr, dtype=np.float32),
                    score=float(t.score),
                )
            )

        tracked.sort(key=lambda t: t.track_id)
        return tracked

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tracker(self, frame_shape: tuple[int, int]) -> None:
        """Initialise the ByteTracker (called on first frame)."""
        try:
            from mmdet.models.trackers import ByteTracker  # type: ignore
        except ImportError:
            # Older mmdet may have it in a different path
            try:
                from mmdet.core.utils.track import ByteTracker  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "ByteTracker not found. Make sure mmdet >= 3.2 is installed "
                    "and 'lap' is available (pip install lap)."
                ) from exc

        self._tracker = ByteTracker(
            track_high_thresh=self._high_thresh,
            track_low_thresh=self._low_thresh,
            new_track_thresh=self._high_thresh,
            track_buffer=self._max_lost,
            match_thresh=0.8,
        )
