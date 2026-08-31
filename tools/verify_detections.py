import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from pathlib import Path

# Add project root to sys.path to import src modules
sys.path.insert(0, 'D:/Downloads/DATN/Skeleton-EAA-Pose')
from src.config_manager import ConfigManager
from src.annotation_reader import get_annotation_reader

def verify_dataset(dataset_name, cfg):
    reader = get_annotation_reader(cfg)
    videos = reader.list_videos()
    detection_dir = Path(cfg.get("detection.output_dir")) / dataset_name
    
    missing_or_incomplete = []
    
    for video_name in videos:
        segments = reader.read(video_name)
        if not segments:
            continue
            
        # Get list of required frame_ids (union of all segments)
        expected_frames = set()
        for seg in segments:
            for f in range(seg.start_frame, seg.end_frame + 1):
                expected_frames.add(f)
                
        jsonl_path = detection_dir / f"{video_name}.jsonl"
        if not jsonl_path.exists():
            missing_or_incomplete.append(video_name)
            continue
            
        # Read actual frame_ids present in the JSONL file
        actual_frames = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "frame_id" in data:
                            actual_frames.add(data["frame_id"])
                    except:
                        pass
        except Exception as e:
            print(f"Error reading {jsonl_path}: {e}", file=sys.stderr)
            missing_or_incomplete.append(video_name)
            continue
            
        # Check if any frame in Annotation is missing from jsonl
        missing_frames = expected_frames - actual_frames
        if missing_frames:
            missing_or_incomplete.append(video_name)
            
    return missing_or_incomplete

def main():
    cfg = ConfigManager('D:/Downloads/ĐATN/Skeleton-EAA-Pose/config.yaml')
    
    # 1. Review PKU
    cfg.set('dataset', 'PKU')
    cfg.set('paths.video_dir', 'D:/Downloads/ĐATN/PKU-MMD/Data_PKUMMD')
    cfg.set('paths.annotation_dir', 'D:/Downloads/ĐATN/PKU-MMD/Label_PKUMMD')
    cfg.set('detection.output_dir', 'D:/Downloads/ĐATN/outputs_detection')
    
    print("Verifying PKU...")
    pku_missing = verify_dataset('PKU', cfg)
    
    # 2. Review TSU
    cfg.set('dataset', 'TSU')
    cfg.set('paths.video_dir', 'D:/Downloads/ĐATN') 
    cfg.set('paths.annotation_dir', 'D:/Downloads/DATN/Annotation_v1.0')
    cfg.set('tsu.event_map', None) 
    
    print("Verifying TSU...")
    tsu_missing = verify_dataset('TSU', cfg)
    
    # Combine and print the results
    all_missing = pku_missing + tsu_missing
    
    if all_missing:
        print("\nList of videos with missing chunks or missing frames detected:")
        output_list = [f'"{name}"' for name in all_missing]
        print(", ".join(output_list))
    else:
        print("\nAll videos have complete detected frames!")

if __name__ == "__main__":
    main()
