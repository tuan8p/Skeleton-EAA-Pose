"""
eaa_pose.datasets.pku
======================
Dataset adapter for PKU-MMD v1 and v2.

File conventions
----------------
* **Label files** (one per video, e.g. ``0016-R.txt``):

    Each line describes one action instance::

        label_id,start_frame,end_frame,confidence

* **Video files**: ``<video_id>.<ext>`` (typically ``.avi`` for v1,
  may vary for v2).  The stem of the video file must match the stem of
  the label file.

* **Actions file** (``.xlsx`` or ``.csv``): columns ``Label`` (int) and ``Action`` (str).
  Used to resolve ``label_id → label_name``.

Usage
-----
    from eaa_pose.datasets.pku import PKUDataset

    dataset = PKUDataset(
        video_dir="RGB_PKUMMDv1",
        segments_dir="out/Label_PKUMMDv1_daily",
        actions_xlsx="out/Actions_daily.csv",
        video_ext=".avi",
    )
    entries = dataset.load()
    for entry in entries:
        print(entry.video_id, len(entry.segments))
"""

from __future__ import annotations

import csv
import warnings
import openpyxl
from pathlib import Path

from .base import ActionSegment, BaseDataset, VideoEntry


class PKUDataset(BaseDataset):
    """Dataset adapter for PKU-MMD (v1 or v2).

    Parameters
    ----------
    video_dir:
        Directory containing video files (``*.avi`` or ``*.mp4``).
    segments_dir:
        Directory containing filtered label ``.txt`` files
        (output of :mod:`eaa_pose.filter_pku_interactions` for v1,
        or the raw label directory for v2).
    actions_xlsx:
        Path to the actions mapping file (``.xlsx`` or ``.csv``):
        ``Label → Action``.
    video_ext:
        Video file extension (default ``.avi``).
    """

    def __init__(
        self,
        video_dir: str | Path,
        segments_dir: str | Path,
        actions_xlsx: str | Path,
        video_ext: str = ".avi",
    ) -> None:
        self._video_dir    = Path(video_dir)
        self._segments_dir = Path(segments_dir)
        self._actions_xlsx = Path(actions_xlsx)
        self._video_ext    = video_ext
        self._entries: list[VideoEntry] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[VideoEntry]:
        """Parse all label files and return matched VideoEntry objects.

        A VideoEntry is created only when BOTH a label file AND the
        corresponding video file exist.  Label files with no matching
        video are skipped with a warning.

        Returns
        -------
        list of :class:`VideoEntry` sorted by ``video_id``.
        """
        if self._entries is not None:
            return self._entries

        if not self._segments_dir.exists():
            warnings.warn(f"PKU: segments_dir does not exist: {self._segments_dir}", stacklevel=2)
            self._entries = []
            return self._entries
        if not self._video_dir.exists():
            warnings.warn(f"PKU: video_dir does not exist: {self._video_dir}", stacklevel=2)
            self._entries = []
            return self._entries

        actions = self._load_actions()
        entries: list[VideoEntry] = []
        video_lookup = self._build_video_lookup()
        label_files = sorted(self._segments_dir.rglob("*.txt"))

        missing_videos: list[str] = []
        empty_segments = 0
        for label_path in label_files:
            video_id = label_path.stem
            video_path = video_lookup.get(video_id.lower())

            if video_path is None:
                missing_videos.append(video_id)
                continue

            segments = self._parse_label_file(label_path, actions)
            if not segments:
                empty_segments += 1
                continue

            entries.append(
                VideoEntry(
                    video_id=video_id,
                    video_path=video_path,
                    segments=tuple(segments),
                )
            )

        self._entries = sorted(entries, key=lambda e: e.video_id)
        if not self._entries:
            warnings.warn(
                "PKU: no matched videos loaded. "
                f"labels_found={len(label_files)}, videos_found={len(video_lookup)}, "
                f"missing_video_matches={len(missing_videos)}, empty_label_files={empty_segments}. "
                f"segments_dir={self._segments_dir}, video_dir={self._video_dir}, "
                f"video_ext={self._video_ext}. "
                f"missing_examples={missing_videos[:5]}",
                stacklevel=2,
            )
        return self._entries

    def __len__(self) -> int:
        return len(self.load())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_video_lookup(self) -> dict[str, Path]:
        """Return lower-case video stem -> video path, scanning recursively."""
        lookup: dict[str, Path] = {}
        ext = self._video_ext.lower()
        for path in sorted(self._video_dir.rglob(f"*{self._video_ext}")):
            if path.suffix.lower() != ext:
                continue
            lookup.setdefault(path.stem.lower(), path)
        return lookup

    def _load_actions(self) -> dict[int, str]:
        """Load label_id → action_name mapping from Actions.xlsx or .csv.

        Returns an empty dict if the file does not exist, allowing the
        adapter to function without action names (label_name will be '').
        """
        if not self._actions_xlsx.exists():
            return {}

        if self._actions_xlsx.suffix.lower() == ".csv":
            return self._load_actions_csv(self._actions_xlsx)

        return self._load_actions_xlsx(self._actions_xlsx)

    @staticmethod
    def _load_actions_csv(path: Path) -> dict[int, str]:
        actions: dict[int, str] = {}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label_raw = row.get("Label")
                action_name = row.get("Action")
                if label_raw is None or action_name is None:
                    continue
                try:
                    actions[int(label_raw)] = str(action_name)
                except (ValueError, TypeError):
                    pass
        return actions

    @staticmethod
    def _load_actions_xlsx(path: Path) -> dict[int, str]:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        actions: dict[int, str] = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            if len(row) < 2:
                continue
            label_id, action_name = row[0], row[1]
            if label_id is not None and action_name is not None:
                try:
                    actions[int(label_id)] = str(action_name)
                except (ValueError, TypeError):
                    pass
        wb.close()
        return actions

    @staticmethod
    def _parse_label_file(
        path: Path,
        actions: dict[int, str],
    ) -> list[ActionSegment]:
        """Parse a PKU label file into a sorted list of ActionSegments.

        Lines that cannot be parsed are silently skipped.

        Parameters
        ----------
        path:
            Path to the label ``.txt`` file.
        actions:
            Mapping ``label_id → action_name`` from Actions.xlsx.

        Returns
        -------
        List of :class:`ActionSegment` sorted by ``start_frame``.
        """
        raw: list[tuple[int, int, int, int]] = []  # (label_id, start, end, conf)

        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                try:
                    label_id    = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame   = int(parts[2])
                except ValueError:
                    continue
                raw.append((label_id, start_frame, end_frame, 0))

        # Sort by start_frame for temporal ordering
        raw.sort(key=lambda x: x[1])

        segments = [
            ActionSegment(
                seg_id=idx,
                label_id=label_id,
                label_name=actions.get(label_id, ""),
                start_frame=start_frame,
                end_frame=end_frame,
            )
            for idx, (label_id, start_frame, end_frame, _) in enumerate(raw)
        ]
        return segments
