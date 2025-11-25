#!/usr/bin/env python3
"""CLI tool for dataset generation.

Generate RenderFormer training datasets.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import bpy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scene_processor.dataset_config import DatasetConfig
from scene_processor.dataset_generator import generate_training_dataset
from scene_processor.objaverse_downloader import download_objaverse_objects_simple

def _load_ids_from_file(path: str) -> list[str]:
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids

def main():
    parser = argparse.ArgumentParser(
        description="Generate RenderFormer training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output_dir", type=str, default="output/dataset",
                       help="Output directory for dataset")
    parser.add_argument("--num_scenes", type=int, default=100,
                       help="Number of scenes to generate")
    parser.add_argument("--num_views", type=int, default=4,
                       help="Number of views per scene")
    parser.add_argument("--template_id", type=int, default=None, choices=[0, 1, 2, 3, None],
                       help="Template scene ID (0-3). If None, randomly selects from 0-3 uniformly for each scene. "
                            "Note: Use JSON config file with [min, max] list to specify a range.")
    
    parser.add_argument("--num_objects", type=int, default=None,
                       help="Number of objects per scene. If None, randomly samples 1-3 objects with 50/30/20 ratio. "
                            "Note: Use JSON config file with [min, max] list to specify a range.")
    parser.add_argument("--num_lights", type=int, default=None,
                       help="Number of lights per scene (max 8). If None, randomly selects 2-4 lights for each scene. "
                            "Note: Use JSON config file with [min, max] list to specify a range.")
    
    parser.add_argument("--resolution", type=int, default=256,
                       help="Rendering resolution (256 for RTX 3090)")
    parser.add_argument("--spp", type=int, default=512,
                       help="Samples per pixel (512 for RTX 3090)")
    parser.add_argument("--max_triangles", type=int, default=2048,
                       help="Maximum number of triangles per scene")
    
    parser.add_argument("--objaverse_root", type=str, default=None,
                       help="Objaverse root directory (if objects are already downloaded)")
    parser.add_argument("--objaverse_object_ids", type=str, nargs='+', default=None,
                       help="Objaverse object IDs to use (if None, uses placeholder)")
    parser.add_argument("--objaverse_ids_file", type=str, default=None,
                       help="Path to a text file containing Objaverse IDs (one per line)")
    
    parser.add_argument("--examples_dir", type=str, default="examples",
                       help="Examples directory path")
    
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (None = CPU cores - 1)")
    
    parser.add_argument("--skip_rendering", action="store_true",
                       help="Skip rendering (for testing)")
    parser.add_argument("--no_gpu_check", action="store_true",
                       help="Disable GPU memory check")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file (overrides command line args)")
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save config to JSON file")
    
    args = parser.parse_args()
    
    if args.config:
        config = DatasetConfig.load(args.config)
    else:
        config = DatasetConfig(
            num_scenes=args.num_scenes,
            num_views_per_scene=args.num_views,
            template_id=args.template_id,
            num_objects=args.num_objects,
            num_lights=args.num_lights,
            resolution=args.resolution,
            spp=args.spp,
            max_triangles=args.max_triangles,
            examples_dir=args.examples_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            skip_rendering=args.skip_rendering,
            gpu_memory_check=not args.no_gpu_check,
            base_seed=args.seed,
            objaverse_root=args.objaverse_root
        )
    
    if args.save_config:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")
    
    objaverse_mesh_paths: list[str] = []
    object_ids: list[str] = []

    # Check if objaverse_mesh_paths is provided in config
    if hasattr(config, 'objaverse_mesh_paths') and config.objaverse_mesh_paths:
        objaverse_mesh_paths = config.objaverse_mesh_paths
        print(f"Using {len(objaverse_mesh_paths)} mesh paths from config")

    # Command line arguments override config
    if args.objaverse_object_ids:
        # If these look like file paths (contain .obj), use as mesh paths
        if any('.obj' in obj_id for obj_id in args.objaverse_object_ids):
            objaverse_mesh_paths = args.objaverse_object_ids
            print(f"Using {len(objaverse_mesh_paths)} mesh paths from command line")
        else:
            object_ids.extend(args.objaverse_object_ids)

    if args.objaverse_ids_file:
        if os.path.exists(args.objaverse_ids_file):
            file_ids = _load_ids_from_file(args.objaverse_ids_file)
            # Check if file contains paths or IDs
            if any('.obj' in fid for fid in file_ids):
                objaverse_mesh_paths.extend(file_ids)
                print(f"Using {len(file_ids)} mesh paths from file")
            else:
                object_ids.extend(file_ids)
        else:
            print(f"Warning: Objaverse IDs file not found: {args.objaverse_ids_file}")

    # Download Objaverse objects if IDs are provided
    if object_ids and not objaverse_mesh_paths:
        if args.objaverse_root or (hasattr(config, 'objaverse_root') and config.objaverse_root):
            objaverse_root = args.objaverse_root or config.objaverse_root
            objaverse_output_dir = os.path.join(config.output_dir, "objaverse_cache")
            downloaded = download_objaverse_objects_simple(
                object_ids=object_ids,
                output_dir=objaverse_output_dir,
                objaverse_root=objaverse_root
            )
            objaverse_mesh_paths = list(downloaded.values())
            print(f"Downloaded {len(objaverse_mesh_paths)} objects from Objaverse")
        else:
            print("Warning: objaverse_root not provided, cannot download Objaverse objects")
    
    if not objaverse_mesh_paths:
        print("Warning: No object mesh paths specified, scenes will have no objects")
        print("  Use --objaverse_object_ids to specify object files, e.g.:")
        print("  --objaverse_object_ids $(find examples/objects -name '*.obj')")
    print("=" * 60)
    print("RenderFormer Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Number of scenes: {config.num_scenes}")
    print(f"Views per scene: {config.num_views_per_scene}")
    print(f"Resolution: {config.resolution}x{config.resolution}")
    print(f"SPP: {config.spp}")
    print(f"Max triangles: {config.max_triangles}")
    print("=" * 60)
    
    metadata = generate_training_dataset(
        num_scenes=config.num_scenes,
        num_views_per_scene=config.num_views_per_scene,
        output_dir=config.output_dir,
        template_id=config.template_id,
        num_objects=config.num_objects,
        num_lights=config.num_lights,
        objaverse_mesh_paths=objaverse_mesh_paths,
        scene_center=config.scene_center,
        scene_bounds=config.scene_bounds,
        ground_level=config.ground_level,
        camera_distance_range=config.camera_distance_range,
        camera_fov_range=config.camera_fov_range,
        light_distance_range=config.light_distance_range,
        light_scale_range=config.light_scale_range,
        light_emission_range=config.light_emission_range,
        examples_dir=config.examples_dir,
        max_triangles=config.max_triangles,
        resolution=config.resolution,
        spp=config.spp,
        base_seed=config.base_seed,
        num_workers=config.num_workers,
        skip_rendering=config.skip_rendering,
        gpu_memory_check=config.gpu_memory_check
    )
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"Total scenes: {metadata['num_scenes']}")
    print(f"Total views: {metadata['num_scenes'] * metadata['num_views_per_scene']}")
    print(f"Metadata saved to: {os.path.join(config.output_dir, 'metadata.json')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
