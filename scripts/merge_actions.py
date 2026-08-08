"""(Tuy chon) Gop cac chunk .npy/.jsonl cua 1 video thanh 1 file duy nhat.

Vi du:
    python scripts/merge_actions.py --output-dir out/skeletons --video 0002-L
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def merge_video(output_dir: Path, video_name: str) -> tuple[Path, Path]:
    npy_files = sorted((output_dir / "chunks" / "npy").glob(f"{video_name}_chunk_*.npy"))
    jsonl_files = sorted((output_dir / "chunks" / "jsonl").glob(f"{video_name}_chunk_*.jsonl"))
    if not npy_files:
        raise FileNotFoundError(f"Khong co chunk nao cho {video_name}")
    arrays = [np.load(p) for p in npy_files]
    merged = np.concatenate(arrays, axis=0)
    out_npy = output_dir / f"{video_name}_all.npy"
    np.save(out_npy, merged)
    out_jsonl = output_dir / f"{video_name}_all.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for p in jsonl_files:
            with open(p, "r", encoding="utf-8") as fin:
                fout.write(fin.read())
    return out_npy, out_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunk outputs per video")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    out_npy, out_jsonl = merge_video(Path(args.output_dir), args.video)
    print(f"Saved: {out_npy}\nSaved: {out_jsonl}")


if __name__ == "__main__":
    main()
