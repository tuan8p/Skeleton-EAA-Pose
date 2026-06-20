"""
eaa_pose.datasets.base
======================
Abstract base class for dataset adapters.

Each adapter provides a uniform interface that the pose pipeline
(``run_pose.py``) uses to iterate over videos and their action segments
without knowing dataset-specific file formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActionSegment:
    """A single annotated action segment within a video.

    Attributes
    ----------
    seg_id:
        Zero-based index of this segment in the video's temporal order.
        Used as the ``<id>`` component in the output filename
        ``<video_name>_act<seg_id>_<label_id>.npy``.
    label_id:
        Integer class label (dataset-specific).
    label_name:
        Human-readable action name (empty string if unavailable).
    start_frame:
        Inclusive start frame index (1-based, as stored in annotation).
    end_frame:
        Inclusive end frame index (1-based).
    """

    seg_id: int
    label_id: int
    label_name: str
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        """Number of frames in the segment (inclusive both ends)."""
        return max(0, self.end_frame - self.start_frame + 1)


@dataclass(frozen=True)
class VideoEntry:
    """A video file paired with its list of action segments.

    Attributes
    ----------
    video_id:
        Unique identifier (typically the filename stem, e.g. ``0016-R``).
    video_path:
        Absolute or relative path to the video file.
    segments:
        Temporally sorted list of :class:`ActionSegment` instances.
    """

    video_id: str
    video_path: Path
    segments: tuple[ActionSegment, ...]


class BaseDataset(ABC):
    """Abstract dataset adapter.

    Subclasses implement :meth:`load` to produce a list of
    :class:`VideoEntry` objects.  The rest of the pipeline is
    dataset-agnostic.
    """

    @abstractmethod
    def load(self) -> list[VideoEntry]:
        """Return all video entries for the dataset.

        Returns
        -------
        list of :class:`VideoEntry`, sorted by ``video_id``.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of videos."""
        ...
