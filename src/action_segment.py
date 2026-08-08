"""Data class ActionSegment + xu ly overlap theo PLAN muc 2."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionSegment:
    video_path: str
    video_name: str
    action_id: int
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    dataset: str = "PKU"
    _adjusted_start: int | None = field(default=None, repr=False)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def resolve_overlap(self, last_processed_frame: int) -> tuple[int, int] | None:
        """Tra ve (start, end) can xu ly sau khi loai phan da xu ly.

        None neu toan bo segment da bi phu boi cac segment truoc.
        """
        start = max(self.start_frame, last_processed_frame + 1)
        if start > self.end_frame:
            return None
        self._adjusted_start = start
        return start, self.end_frame

    @property
    def adjusted_start(self) -> int:
        return self._adjusted_start if self._adjusted_start is not None else self.start_frame
