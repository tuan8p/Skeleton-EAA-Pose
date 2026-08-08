"""CLI chay pipeline trich xuat skeleton.

Vi du:
    python scripts/run_extraction.py --config config.yaml --start 0 --end 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline_orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeleton extraction pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", choices=["PKU", "TSU"], default=None)
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Gioi han so segment/video (test nhanh tren CPU)")
    args = parser.parse_args()

    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.max_segments is not None:
        overrides["runtime.max_segments_per_video"] = args.max_segments

    from src.config_manager import ConfigManager
    cfg = ConfigManager(args.config, overrides=overrides)
    orch = PipelineOrchestrator(cfg=cfg)
    orch.run_batch(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
