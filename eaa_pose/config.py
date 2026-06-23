"""
eaa_pose.config
===============
Config loader with three-level precedence:

    base.yaml  <  dataset_config.yaml  <  explicit CLI args

Usage
-----
    from eaa_pose.config import PipelineConfig

    # Load from YAML only
    cfg = PipelineConfig.load("configs/pku_v1.yaml")

    # Load and overlay CLI args (argparse Namespace)
    cfg = PipelineConfig.load("configs/pku_v1.yaml", cli_args=args)

    # Access values like a dict
    print(cfg["filter"]["min_frame_ratio"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Mapping: argparse dest names  →  nested config keys (dot-separated path)
# Only entries in this table are eligible for CLI override.
# ---------------------------------------------------------------------------
_CLI_TO_CONFIG: dict[str, str] = {
    # filter args
    "skeleton_dir":     "filter.skeleton_dir",
    "label_dir":        "filter.label_dir",
    "src_actions_xlsx": "filter.src_actions_xlsx",
    "out_label_dir":    "filter.out_label_dir",
    "out_skeleton_dir": "filter.out_skeleton_dir",
    "out_actions_csv":  "filter.out_actions_csv",
    # pose pipeline args
    "video_dir":       "video_dir",
    "segments_dir":    "segments_dir",
    "actions_xlsx":    "actions_xlsx",
    "out_dir":         "out_dir",
    "device":          "detector.device",
    # Note: estimator.device is resolved in PosePipeline._build_estimator()
    # by falling back to detector.device, so --device covers both.
    "tracking_model":  "tracking.model",
    "tracking_tracker": "tracking.tracker",
    "num_persons":     "output.num_persons",
    # smoke-test args (stored at top level)
    "smoke":           "smoke",
    "max_videos":      "max_videos",
    "max_frames":      "max_frames",
    "limit":           "limit",
    "start_index":     "start_index",
    "end_index":       "end_index",
    "all_videos":      "all",
    "dry_run":         "dry_run",
    "track_qc_max_interp_gap": "track_qc.max_interp_gap",
    "track_qc_max_interp_anchor_distance": "track_qc.max_interp_anchor_distance",
    "track_qc_no_one_sided": "track_qc.no_one_sided",
}


class PipelineConfig:
    """Immutable configuration container.

    Stores merged config as a nested dict and exposes dict-like access.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        dataset_config_path: str | Path,
        cli_args=None,
    ) -> "PipelineConfig":
        """Load and merge configs with full override chain.

        Steps:
        1. Load base.yaml (sibling of the dataset config).
        2. Deep-merge the dataset config on top of base.
        3. If ``cli_args`` is provided, overlay every non-None
           argparse value that appears in ``_CLI_TO_CONFIG``.

        Parameters
        ----------
        dataset_config_path:
            Path to the per-dataset YAML (e.g. ``configs/pku_v1.yaml``).
        cli_args:
            Optional argparse ``Namespace``.  Only keys present in
            ``_CLI_TO_CONFIG`` are considered; ``None`` values are skipped
            so that absent CLI flags do not clobber the config.

        Returns
        -------
        PipelineConfig
        """
        dataset_path = Path(dataset_config_path)
        dataset_raw = cls._load_yaml(dataset_path)

        # Resolve base.yaml relative to the dataset config file
        base_name = dataset_raw.pop("extends", "base.yaml")
        base_path = dataset_path.parent / base_name
        base_raw = cls._load_yaml(base_path)

        # Merge: base first, dataset on top
        merged = cls._deep_merge(base_raw, dataset_raw)

        # CLI override
        if cli_args is not None:
            merged = cls._apply_cli_args(merged, cli_args)

        return cls(merged)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot-separated key.

        Example: ``cfg.get("filter.min_frame_ratio", 0.1)``
        """
        return self._get_nested(self._data, key.split("."), default)

    def __getitem__(self, key: str) -> Any:
        parts = key.split(".")
        val = self._get_nested(self._data, parts, _MISSING)
        if val is _MISSING:
            raise KeyError(key)
        return val

    def __contains__(self, key: str) -> bool:
        return self._get_nested(self._data, key.split("."), _MISSING) is not _MISSING

    def as_dict(self) -> dict:
        """Return a deep copy of the underlying dict."""
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        """Recursively merge ``override`` into a copy of ``base``.

        - Dicts are merged recursively.
        - All other types in ``override`` replace those in ``base``.
        """
        result = copy.deepcopy(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = cls._deep_merge(result[key], val)
            else:
                result[key] = copy.deepcopy(val)
        return result

    @classmethod
    def _apply_cli_args(cls, cfg: dict, cli_args) -> dict:
        """Overlay non-None CLI argument values onto the merged config.

        Only keys listed in ``_CLI_TO_CONFIG`` are considered.
        """
        cfg = copy.deepcopy(cfg)
        ns = vars(cli_args) if not isinstance(cli_args, dict) else cli_args
        for dest, dotted_key in _CLI_TO_CONFIG.items():
            value = ns.get(dest)
            if value is None:
                continue
            cls._set_nested(cfg, dotted_key.split("."), value)
        return cfg

    @staticmethod
    def _get_nested(data: dict, keys: list[str], default: Any) -> Any:
        cur = data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def _set_nested(data: dict, keys: list[str], value: Any) -> None:
        """Set a value in a nested dict, creating intermediate dicts as needed."""
        cur = data
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value


# Sentinel for missing keys
class _MissingType:
    pass

_MISSING = _MissingType()
