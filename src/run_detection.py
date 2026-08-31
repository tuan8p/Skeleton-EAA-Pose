"""CLI runs pipeline detection independently.

For example:
    python src/run_detection.py --config config.local.yaml --dataset PKU --start 0 --end 10
    python src/run_detection.py --config config.local-tsu.yaml --dataset TSU --start 0 --end 5
    python src/run_detection.py --config config.local.yaml --dataset PKU --start 0 --end 3 --max-segments 3
    python src/run_detection.py --config config.local.yaml --dataset PKU --start 0 --end 1 --max-chunks 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_manager import ConfigManager
from src.detection_pipeline import DetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skeleton-EAA-Pose: standalone detection pipeline")
    parser.add_argument("--config", default="config.yaml",
                        help="YAML config file path")
    parser.add_argument("--dataset", choices=["PKU", "TSU"], default=None,
                        help="Override dataset trong config (PKU|TSU)")
    parser.add_argument("--start", type=int, default=0,
                        help="Index videos starting in list (0-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="End video index (exclusive), default = end")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Limit number of segments/video (quick test on CPU)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit number of chunks/video (quick test on CPU)")
    args = parser.parse_args()

    overrides: dict = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.max_segments is not None:
        overrides["runtime.max_segments_per_video"] = args.max_segments
    if args.max_chunks is not None:
        overrides["runtime.max_chunks_per_video"] = args.max_chunks

    cfg = ConfigManager(args.config, overrides=overrides)
    pipeline = DetectionPipeline(cfg=cfg)
    pipeline.run_batch(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
