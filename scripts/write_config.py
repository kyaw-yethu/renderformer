import os
import json
import argparse

def collect_obj_paths(parent_dir):
    """
    Walk `parent_dir` and collect all .obj file paths (relative to cwd).
    """
    obj_paths = []
    for dirpath, _, filenames in os.walk(parent_dir):
        for fname in filenames:
            if fname.lower().endswith(".obj"):
                if 'piece.' in fname:
                    continue  # Skip files with 'piece.' in the name
                full_path = os.path.join(dirpath, fname)
                # Use forward slashes for consistency with your example
                obj_paths.append(full_path.replace(os.sep, "/"))
    obj_paths.sort()
    return obj_paths

def main():
    parser = argparse.ArgumentParser(
        description="Populate objaverse_mesh_paths in a JSON config with all .obj files under a parent directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/configs/medium.json",
        help="Path to the JSON config file to modify.",
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="examples/objects",
        help="Parent directory containing .obj files (subfolders will be scanned).",
    )
    args = parser.parse_args()

    # Collect .obj paths
    mesh_paths = collect_obj_paths(args.parent_dir)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Overwrite objaverse_mesh_paths
    cfg["objaverse_mesh_paths"] = mesh_paths

    # Save config back
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"Updated {args.config} with {len(mesh_paths)} .obj paths.")

if __name__ == "__main__":
    main()
