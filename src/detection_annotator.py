"""Read action annotations for PKU/TSU.
PKU : Label_PKUMMD/<video>.txt, each line: action_id,start_frame,end_frame,confidence
      NO action 'nothing' - only specific actions in each line.
TSU : annotation_dir/**/<video>.csv (event, start_frame, end_frame)
      Only processes events in event_mapping.csv (fixed 53 events).
      Path mapping: config key tsu.event_mapping_path
"""
from __future__ import annotations

import csv
from pathlib import Path

from .action_segment import ActionSegment
from .config_manager import ConfigManager


class PKUDetectionAnnotator:
    def __init__(self, cfg: ConfigManager):
        self.video_dir = Path(cfg.get("paths.video_dir"))
        self.annotation_dir = Path(cfg.get("paths.annotation_dir"))
        self.video_ext: str = cfg.get("pku.video_ext", ".avi")
        self.label_ext: str = cfg.get("pku.label_ext", ".txt")

    def list_videos(self) -> list[str]:
        names = [p.stem for p in sorted(self.annotation_dir.glob(f"*{self.label_ext}"))]
        return [n for n in names if (self.video_dir / f"{n}{self.video_ext}").exists()]

    def read(self, video_name: str) -> list[ActionSegment]:
        label_file = self.annotation_dir / f"{video_name}{self.label_ext}"
        segments: list[ActionSegment] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                action_id = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                conf = float(parts[3]) if len(parts) > 3 else 1.0
                segments.append(ActionSegment(
                    video_path=str(self.video_dir / f"{video_name}{self.video_ext}"),
                    video_name=video_name,
                    action_id=action_id,
                    start_frame=start,
                    end_frame=end,
                    confidence=conf,
                    dataset="PKU",
                ))
        segments.sort(key=lambda s: s.start_frame)
        return segments


class TSUDetectionAnnotator:
    def __init__(self, cfg: ConfigManager):
        self.video_dir = Path(cfg.get("paths.video_dir"))
        self.annotation_dir = Path(cfg.get("paths.annotation_dir"))
        self.video_ext: str = cfg.get("tsu.video_ext", ".mp4")
        mapping_path = Path(cfg.get("tsu.event_mapping_path", "event_mapping.csv"))
        self._event_map: dict[str, int] = self._load_mapping(mapping_path)
        # Column config: string name if has_header=True, index if False
        self.has_header: bool = bool(cfg.get("tsu.has_header", True))
        self.event_col = cfg.get("tsu.event_column", "event")
        self.start_col = cfg.get("tsu.start_column", "start_frame")
        self.end_col = cfg.get("tsu.end_column", "end_frame")

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, int]:
        """Load event_mapping.csv (columns: id, event_name) → {event_name: id}.

        Use utf-8-sig to automatically strip BOM (\ufeff) - Windows Excel often adds BOM.
        """
        mapping: dict[str, int] = {}
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("event_name", "").strip()
                eid = row.get("id", "").strip()
                if name and eid:
                    mapping[name] = int(eid)
        return mapping

    def list_videos(self) -> list[str]:
        names = sorted({p.stem for p in self.annotation_dir.glob("**/*.csv")})
        return [n for n in names if (self.video_dir / f"{n}{self.video_ext}").exists()]

    def read(self, video_name: str) -> list[ActionSegment]:
        matches = list(self.annotation_dir.glob(f"**/{video_name}.csv"))
        if not matches:
            return []
        segments: list[ActionSegment] = []
        with open(matches[0], "r", encoding="utf-8", newline="") as f:
            if self.has_header:
                reader = csv.DictReader(f)
                # Supports both column names (string) and index (int)
                event_key = (self.event_col if isinstance(self.event_col, str)
                             else None)
                start_key = (self.start_col if isinstance(self.start_col, str)
                             else None)
                end_key = (self.end_col if isinstance(self.end_col, str)
                           else None)
                for row in reader:
                    fieldnames = reader.fieldnames or []
                    evt = row.get(event_key or fieldnames[self.event_col], "").strip()
                    if evt not in self._event_map:
                        continue
                    action_id = self._event_map[evt]
                    try:
                        start = int(row.get(start_key or fieldnames[self.start_col], 0))
                        end = int(row.get(end_key or fieldnames[self.end_col], 0))
                    except (ValueError, IndexError, TypeError):
                        continue
                    if end < start:
                        continue
                    segments.append(ActionSegment(
                        video_path=str(self.video_dir / f"{video_name}{self.video_ext}"),
                        video_name=video_name,
                        action_id=action_id,
                        start_frame=start,
                        end_frame=end,
                        confidence=1.0,
                        dataset="TSU",
                    ))
            else:
                # No header: read by index
                reader_raw = csv.reader(f)
                for row in reader_raw:
                    if len(row) < 3:
                        continue
                    evt = row[int(self.event_col)].strip()
                    if evt not in self._event_map:
                        continue
                    action_id = self._event_map[evt]
                    try:
                        start = int(row[int(self.start_col)])
                        end = int(row[int(self.end_col)])
                    except (ValueError, IndexError):
                        continue
                    if end < start:
                        continue
                    segments.append(ActionSegment(
                        video_path=str(self.video_dir / f"{video_name}{self.video_ext}"),
                        video_name=video_name,
                        action_id=action_id,
                        start_frame=start,
                        end_frame=end,
                        confidence=1.0,
                        dataset="TSU",
                    ))
        segments.sort(key=lambda s: s.start_frame)
        return segments


def get_detection_annotator(cfg: ConfigManager):
    dataset = str(cfg.get("dataset", "PKU")).upper()
    if dataset == "PKU":
        return PKUDetectionAnnotator(cfg)
    if dataset == "TSU":
        return TSUDetectionAnnotator(cfg)
    raise ValueError(f"Unsupported dataset: {dataset}")
