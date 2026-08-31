"""Read PKU-MMD Phase 1 and TSU annotations, return list of ActionSegment.

PKU : Label_PKUMMD/<video>.txt, moi dong: action_id,start_frame,end_frame,confidence
TSU : annotation_dir/**/<video>.csv (cau truc phang hoac theo person), moi dong:
      event,start_frame,end_frame — event la TEN hanh dong (ke ca sub-action,
      coi la label rieng biet). Ten event -> action_id qua tsu.event_map trong
      config; neu de trong thi tu build (sorted unique -> 1..N) va luu
      tsu_event_map.json vao output_dir de nhat quan giua cac lan chay.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .action_segment import ActionSegment
from .config_manager import ConfigManager


class PKUAnnotationReader:
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
                action_id, start, end = int(parts[0]), int(parts[1]), int(parts[2])
                conf = float(parts[3]) if len(parts) > 3 else 1.0
                segments.append(ActionSegment(
                    video_path=str(self.video_dir / f"{video_name}{self.video_ext}"),
                    video_name=video_name, action_id=action_id,
                    start_frame=start, end_frame=end, confidence=conf, dataset="PKU",
                ))
        segments.sort(key=lambda s: s.start_frame)
        return segments


class TSUAnnotationReader:
    def __init__(self, cfg: ConfigManager):
        self.video_dir = Path(cfg.get("paths.video_dir"))
        self.annotation_dir = Path(cfg.get("paths.annotation_dir"))
        self.video_ext: str = cfg.get("tsu.video_ext", ".mp4")
        self.has_header: bool = bool(cfg.get("tsu.has_header", True))
        self.col_event: int = int(cfg.get("tsu.event_column", 0))
        self.col_start: int = int(cfg.get("tsu.start_column", 1))
        self.col_end: int = int(cfg.get("tsu.end_column", 2))
        self._event_map: dict[str, int] | None = cfg.get("tsu.event_map") or None
        self._map_file = Path(cfg.get("paths.output_dir")) / "tsu_event_map.json"

    @property
    def event_map(self) -> dict[str, int]:
        if self._event_map is None:
            self._event_map = self._load_or_build_event_map()
        return self._event_map

    def _load_or_build_event_map(self) -> dict[str, int]:
        if self._map_file.exists():
            with open(self._map_file, "r", encoding="utf-8") as f:
                return json.load(f)
        events: set[str] = set()
        for csv_path in self.annotation_dir.glob("**/*.csv"):
            for raw in self._iter_events(csv_path):
                events.add(raw)
        event_map = {name: i + 1 for i, name in enumerate(sorted(events))}
        self._map_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._map_file, "w", encoding="utf-8") as f:
            json.dump(event_map, f, indent=1, ensure_ascii=False)
        return event_map

    def _iter_events(self, csv_path: Path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            if self.has_header:
                next(reader, None)
            for row in reader:
                if len(row) > max(self.col_event, self.col_start, self.col_end):
                    yield row[self.col_event].strip()

    def list_videos(self) -> list[str]:
        names = sorted({p.stem for p in self.annotation_dir.glob("**/*.csv")})
        return [n for n in names if (self.video_dir / f"{n}{self.video_ext}").exists()]

    def read(self, video_name: str) -> list[ActionSegment]:
        matches = [p for p in self.annotation_dir.glob(f"**/{video_name}.csv")]
        if not matches:
            return []
        segments: list[ActionSegment] = []
        with open(matches[0], "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            if self.has_header:
                next(reader, None)
            for row in reader:
                if len(row) <= max(self.col_event, self.col_start, self.col_end):
                    continue
                event = row[self.col_event].strip()
                if event not in self.event_map:
                    continue
                segments.append(ActionSegment(
                    video_path=str(self.video_dir / f"{video_name}{self.video_ext}"),
                    video_name=video_name,
                    action_id=int(self.event_map[event]),
                    start_frame=int(row[self.col_start]), end_frame=int(row[self.col_end]),
                    confidence=1.0, dataset="TSU",
                ))
        segments.sort(key=lambda s: s.start_frame)
        return segments


def get_annotation_reader(cfg: ConfigManager):
    dataset = str(cfg.get("dataset", "PKU")).upper()
    if dataset == "PKU":
        return PKUAnnotationReader(cfg)
    if dataset == "TSU":
        return TSUAnnotationReader(cfg)
    raise ValueError(f"Dataset khong ho tro: {dataset}")
