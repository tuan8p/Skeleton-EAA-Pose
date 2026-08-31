"""Manage progress.json for pipeline resume."""
from __future__ import annotations

import json
import threading
from pathlib import Path


class ProgressManager:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def is_chunk_done(self, video_name: str, chunk_index: int) -> bool:
        with self._lock:
            return chunk_index in self._data.get(video_name, {}).get("chunks_done", [])

    def is_video_done(self, video_name: str) -> bool:
        with self._lock:
            return bool(self._data.get(video_name, {}).get("finished", False))

    def mark_chunk_done(self, video_name: str, chunk_index: int, total_chunks: int) -> None:
        with self._lock:
            entry = self._data.setdefault(video_name, {"chunks_done": [], "total_chunks": total_chunks,
                                                       "finished": False})
            if chunk_index not in entry["chunks_done"]:
                entry["chunks_done"].append(chunk_index)
                entry["chunks_done"].sort()
            entry["total_chunks"] = total_chunks
        self.save()

    def mark_video_done(self, video_name: str) -> None:
        with self._lock:
            self._data.setdefault(video_name, {})["finished"] = True
        self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=1)
