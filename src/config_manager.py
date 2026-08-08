"""Doc/ghi file config YAML, cung cap tham so cho toan pipeline."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    def __init__(self, config_path: str | Path | None = None, overrides: dict | None = None):
        self._cfg: dict = {}
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
        if overrides:
            self.apply_overrides(overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """Truy cap theo dot-path, vi du: 'mediapipe.model_complexity'."""
        node = self._cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        node = self._cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def apply_overrides(self, overrides: dict) -> None:
        """Ghi de theo dot-path: {'mediapipe.model_complexity': 2}."""
        for key, value in overrides.items():
            self.set(key, value)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._cfg)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._cfg, f, allow_unicode=True, sort_keys=False)
