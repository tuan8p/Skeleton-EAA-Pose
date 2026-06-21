"""
eaa_pose.datasets.tsu
======================
Dataset adapter for Toyota Smarthome Untrimmed (TSU).

Annotation structure
--------------------
``Annotation_v1.0/<subject_id>/<sample>.csv``

Each CSV file corresponds to one video sample.  The CSV has columns::

    event, start_frame, end_frame

where ``event`` is the action class name (string).

The matching video file lives in::

    Videos_mp4/<sample>.mp4   (or ``<subject_id>_<sample>.mp4``)

Because TSU video filenames vary by release, the adapter tries multiple
naming conventions and warns when a match cannot be found.

Usage
-----
    from eaa_pose.datasets.tsu import TSUDataset

    dataset = TSUDataset(
        video_dir="/path/to/Videos_mp4",
        segments_dir="/path/to/Annotation_v1.0",
        video_ext=".mp4",
    )
    entries = dataset.load()
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

from .base import ActionSegment, BaseDataset, VideoEntry


class TSUDataset(BaseDataset):
    """Dataset adapter for Toyota Smarthome Untrimmed (TSU).

    Parameters
    ----------
    video_dir:
        Directory containing ``.mp4`` video files.
    segments_dir:
        Root of the annotation directory
        (``Annotation_v1.0/<subject_id>/<sample>.csv``).
    video_ext:
        Video file extension (default ``.mp4``).
    """

    def __init__(
        self,
        video_dir: str | Path,
        segments_dir: str | Path,
        video_ext: str = ".mp4",
    ) -> None:
        self._video_dir    = Path(video_dir)
        self._segments_dir = Path(segments_dir)
        self._video_ext    = video_ext
        self._entries: list[VideoEntry] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[VideoEntry]:
        """Scan all subject annotation folders and return VideoEntry objects.

        Returns
        -------
        list of :class:`VideoEntry` sorted by ``video_id``.
        """
        if self._entries is not None:
            return self._entries

        # Build a fast lookup: video_stem → Path for all mp4 files
        video_lookup = self._build_video_lookup()

        entries: list[VideoEntry] = []

        for subject_dir in sorted(self._segments_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name

            for csv_path in sorted(subject_dir.glob("*.csv")):
                video_id = csv_path.stem   # e.g. "P01_R00_S01"
                video_path = self._resolve_video(video_id, subject_id, video_lookup)

                if video_path is None:
                    warnings.warn(
                        f"TSU: no video found for annotation '{csv_path}'. "
                        f"Tried stems: '{video_id}', '{subject_id}_{video_id}'. "
                        "Skipping.",
                        stacklevel=2,
                    )
                    continue

                segments = self._parse_csv(csv_path)
                if not segments:
                    continue

                entries.append(
                    VideoEntry(
                        video_id=video_id,
                        video_path=video_path,
                        segments=tuple(segments),
                    )
                )

        self._entries = sorted(entries, key=lambda e: e.video_id)
        return self._entries

    def __len__(self) -> int:
        return len(self.load())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_video_lookup(self) -> dict[str, Path]:
        """Return a dict mapping ``lower(video_stem) → video_path``."""
        lookup: dict[str, Path] = {}
        ext = self._video_ext.lower()
        for p in self._video_dir.rglob(f"*{self._video_ext}"):
            if p.suffix.lower() == ext:
                lookup[p.stem.lower()] = p
        return lookup

    def _resolve_video(
        self,
        video_id: str,
        subject_id: str,
        lookup: dict[str, Path],
    ) -> Path | None:
        """Try multiple naming conventions to locate the video file.

        Conventions tried (in order):
        1. ``<video_id><ext>``              e.g. ``P01_R00_S01.mp4``
        2. ``<subject_id>_<video_id><ext>`` e.g. ``P01_P01_R00_S01.mp4``
        3. Prefix match on video_id
        """
        for candidate in (video_id.lower(), f"{subject_id}_{video_id}".lower()):
            if candidate in lookup:
                return lookup[candidate]

        # Prefix match as last resort
        prefix = video_id.lower()
        for stem, path in lookup.items():
            if stem.startswith(prefix):
                return path

        return None

    @staticmethod
    def _parse_csv(path: Path) -> list[ActionSegment]:
        """Parse a TSU annotation CSV into sorted ActionSegments.

        Expected columns: ``event``, ``start_frame``, ``end_frame``
        (case-insensitive; extra columns are ignored).

        Parameters
        ----------
        path: Path to the annotation CSV file.

        Returns
        -------
        List of :class:`ActionSegment` sorted by ``start_frame``.
        """
        raw: list[tuple[str, int, int]] = []

        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []

            # Normalise column names to lower-case for robustness
            norm = {k.strip().lower(): k for k in reader.fieldnames if k}

            event_col = norm.get("event")
            start_col = norm.get("start_frame")
            end_col   = norm.get("end_frame")

            if not all([event_col, start_col, end_col]):
                warnings.warn(
                    f"TSU CSV '{path}' is missing required columns "
                    f"(event, start_frame, end_frame). Found: {list(reader.fieldnames)}",
                    stacklevel=2,
                )
                return []

            for row in reader:
                try:
                    event       = row[event_col].strip()
                    start_frame = int(row[start_col])
                    end_frame   = int(row[end_col])
                except (ValueError, KeyError):
                    continue
                if event:
                    raw.append((event, start_frame, end_frame))

        # Sort by start_frame for temporal ordering
        raw.sort(key=lambda x: x[1])

        # Build a per-dataset label_id by hashing the event name to a
        # stable integer.  TSU does not have a numeric label scheme in
        # the annotation files.
        seen: dict[str, int] = {}
        segments: list[ActionSegment] = []
        for idx, (event, start_frame, end_frame) in enumerate(raw):
            if event not in seen:
                seen[event] = len(seen) + 1   # 1-based
            segments.append(
                ActionSegment(
                    seg_id=idx,
                    label_id=seen[event],
                    label_name=event,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        return segments
