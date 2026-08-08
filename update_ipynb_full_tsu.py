import json

nb_path = r"d:\Downloads\ĐATN\Skeleton-EAA-Pose\notebooks\extract_skeleton.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        in_settings = False
        for line in source:
            if "SETTINGS = {" in line:
                in_settings = True
                new_source.append(line)
                continue
            if in_settings and "}" in line and not line.strip().startswith("#"):
                in_settings = False
                # Remove the previously added single line if it exists
                if new_source[-1].strip().startswith('"tsu.event_mapping_path"'):
                    new_source.pop()
                
                # Add all TSU configs
                new_source.append('    # --- TSU Specific Config ---\n')
                new_source.append('    "tsu.video_ext": ".mp4",\n')
                new_source.append('    "tsu.has_header": True,\n')
                new_source.append('    "tsu.event_column": 0,\n')
                new_source.append('    "tsu.start_column": 1,\n')
                new_source.append('    "tsu.end_column": 2,\n')
                new_source.append('    "tsu.event_map": {},          # {"event_name": action_id}\n')
                new_source.append('    "tsu.event_mapping_path": "event_mapping.csv", # File anh xa event TSU co dinh\n')
                new_source.append(line)
                continue
            
            # Skip the single line if it was already added somewhere else in the block
            if in_settings and '"tsu.event_mapping_path"' in line:
                continue
                
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
