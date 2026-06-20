"""
eaa_pose.io.sample_writer
==========================
Per-action-sample serializer.

Each action segment is saved as a single ``.npy`` file with the naming
convention::

    <out_dir>/<video_name>_act<seg_id>_<label_id>.npy

Array shape and layout
----------------------
``(3, T, 25, M)``

    3   = coordinate channels [x, y, z]
    T   = number of frames in the segment
    25  = NTU-120 joints
    M   = number of persons (typically 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SampleWriter:
    """Writes per-action-segment skeleton arrays to ``.npy`` files.

    Parameters
    ----------
    out_dir:
        Directory where output files are written.  Created if needed.
    dtype:
        Numpy dtype for the saved array (default ``float32``).
    overwrite:
        If False (default), skip files that already exist.
    """

    def __init__(
        self,
        out_dir: str | Path,
        dtype: str = "float32",
        overwrite: bool = False,
    ) -> None:
        self._out_dir  = Path(out_dir)
        self._dtype    = np.dtype(dtype)
        self._overwrite = overwrite
        self._out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        array: np.ndarray,
        video_name: str,
        seg_id: int,
        label_id: int,
    ) -> Path:
        """Save a single action-segment array.

        Parameters
        ----------
        array:
            Internal QC'd skeleton array of shape ``(T, M, 25, 6)``.
        video_name:
            Video stem used in the filename (e.g. ``'0063-R'``).
        seg_id:
            Segment index within the video (zero-based).
        label_id:
            Integer class label used in the filename.

        Returns
        -------
        Path of the written file.
        """
        filename = self.filename(video_name, seg_id, label_id)
        out_path = self._out_dir / filename

        if out_path.exists() and not self._overwrite:
            return out_path

        np.save(out_path, self.to_skateformer(array).astype(self._dtype))
        return out_path

    def output_path(self, video_name: str, seg_id: int, label_id: int) -> Path:
        """Return the expected output path without writing anything."""
        return self._out_dir / self.filename(video_name, seg_id, label_id)

    @staticmethod
    def filename(video_name: str, seg_id: int, label_id: int) -> str:
        """Return the sample filename."""
        return f"{video_name}_act{seg_id}_{label_id}.npy"

    @staticmethod
    def to_skateformer(array: np.ndarray) -> np.ndarray:
        """Convert internal ``(T, M, 25, 6)`` arrays to ``(3, T, 25, M)``."""
        coords = array[..., :3]          # (T, M, 25, 3)
        return np.transpose(coords, (3, 0, 2, 1))

    def write_metadata(
        self,
        metadata: dict[str, Any],
        filename: str = "metadata.json",
        overwrite: bool = True,
    ) -> Path:
        """Save dataset-level metadata as JSON."""
        out_path = self._out_dir / filename

        if out_path.exists() and not overwrite and not self._overwrite:
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        return out_path
