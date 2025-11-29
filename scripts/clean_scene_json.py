import json
import os
from pathlib import Path

METADATA_PATH = Path("output/dataset_medium/metadata.json")  # change if needed

def main():
    # Load metadata
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    scenes = metadata.get("scenes", [])
    kept_scenes = []
    removed_count = 0

    for scene in scenes:
        h5_path = scene.get("h5_path", None)

        # If h5_path is null -> remove scene entry and delete its json_path file
        if h5_path is None:
            json_path = scene.get("json_path")
            if json_path:
                json_file = Path(json_path)
                if json_file.exists():
                    os.remove(json_file)
                    print(f"Deleted JSON file: {json_file}")
                else:
                    print(f"JSON file not found (skipped): {json_file}")
            else:
                print(f"Scene {scene.get('scene_name')} has null h5_path and no json_path")
            removed_count += 1
        else:
            kept_scenes.append(scene)

    metadata["scenes"] = kept_scenes

    # Optionally keep num_scenes in sync if it exists
    if "num_scenes" in metadata:
        metadata["num_scenes"] = len(kept_scenes)

    # Write cleaned metadata
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Removed {removed_count} scenes with null h5_path")
    print(f"Remaining scenes: {len(kept_scenes)}")
    print(f"Updated metadata written to: {METADATA_PATH}")

if __name__ == "__main__":
    main()
