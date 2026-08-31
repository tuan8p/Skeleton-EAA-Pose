"""Split skeleton chunks into per-action files for a video.

Doc : output_dir/chunks/{npy,jsonl,meta}/<video>_chunk_NNN.*
Ghi : output_dir/actions/{npy,jsonl}/<video>_NNN.*
      (<video>_000 = action dau tien cua video)

Vung OVERLAP giua 2 action duoc ghi vao CA 2 file action (extract 1 lan,
dung chung — dung co che moi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def split_video(output_dir: Path, video_name: str) -> list[tuple[Path, int, int]]:
    """Return list of (action_file, action_id, frame_count)."""
    chunk_npy_dir = output_dir / "chunks" / "npy"
    chunk_jsonl_dir = output_dir / "chunks" / "jsonl"
    chunk_meta_dir = output_dir / "chunks" / "meta"
    action_npy_dir = output_dir / "actions" / "npy"
    action_jsonl_dir = output_dir / "actions" / "jsonl"

    # gather original segments in video order (chunks in sequence)
    segments: list[dict] = []
    chunk_data: list[dict] = []
    for meta_path in sorted(chunk_meta_dir.glob(f"{video_name}_chunk_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ci = meta["chunk_index"]
        npy = np.load(chunk_npy_dir / f"{video_name}_chunk_{ci:03d}.npy")
        rows = [json.loads(l) for l in
                (chunk_jsonl_dir / f"{video_name}_chunk_{ci:03d}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        fid2idx = {r["frame_id"]: i for i, r in enumerate(rows)}
        chunk_data.append({"npy": npy, "rows": rows, "fid2idx": fid2idx,
                           "segments": meta["segments"]})
        segments.extend(meta["segments"])
    if not chunk_data:
        raise FileNotFoundError(f"Khong co chunk meta nao cho {video_name}")
    segments.sort(key=lambda s: s["start"])

    action_npy_dir.mkdir(parents=True, exist_ok=True)
    action_jsonl_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, seg in enumerate(segments):
        out_npy = action_npy_dir / f"{video_name}_{idx:03d}.npy"
        out_jsonl = action_jsonl_dir / f"{video_name}_{idx:03d}.jsonl"
        wrote = False
        for cd in chunk_data:
            if seg["start"] in cd["fid2idx"] and seg["end"] in cd["fid2idx"]:
                i0, i1 = cd["fid2idx"][seg["start"]], cd["fid2idx"][seg["end"]]
                np.save(out_npy, cd["npy"][i0:i1 + 1].astype(np.float32, copy=False))
                with open(out_jsonl, "w", encoding="utf-8") as f:
                    for row in cd["rows"][i0:i1 + 1]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.append((out_npy, seg["action_id"], i1 - i0 + 1))
                wrote = True
                break
        if not wrote:
            print(f"CANH BAO: segment {seg} khong tim thay du frame trong chunk")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Split chunk skeletons into per-action files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    results = split_video(Path(args.output_dir), args.video)
    for path, action_id, n_frames in results:
        print(f"{path.name}: action={action_id}, frames={n_frames}")


if __name__ == "__main__":
    main()
