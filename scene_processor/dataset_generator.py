"""Training dataset generation pipeline.

Integrated pipeline from scene generation to HDF5+GT images.
"""
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from tqdm import tqdm
from .scene_config import SceneConfig
from .batch_scene_generator import generate_scene_batch
from .batch_renderer import render_scene_batch

def generate_training_dataset(
    num_scenes: int,
    num_views_per_scene: int,
    output_dir: str,
    template_id: Optional[Union[int, Tuple[int, int]]] = None,
    num_objects: Optional[Union[int, Tuple[int, int]]] = None,
    num_lights: Optional[Union[int, Tuple[int, int]]] = None,
    objaverse_mesh_paths: List[str] = None,
    scene_center: tuple = (0.0, 0.0, 0.0),
    scene_bounds: tuple = (-0.5, 0.5),
    ground_level: float = -0.5,
    camera_distance_range: tuple = (1.5, 2.0),
    camera_fov_range: tuple = (30.0, 60.0),
    light_distance_range: tuple = (2.1, 2.7),
    light_scale_range: tuple = (2.0, 2.5),
    light_emission_range: tuple = (2500.0, 5000.0),
    examples_dir: str = "examples",
    resolution: int = 256,
    spp: int = 512,
    base_seed: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_rendering: bool = False,
    gpu_memory_check: bool = True
) -> Dict:
    """Generate training dataset.
    
    Args:
        num_scenes: Number of scenes to generate
        num_views_per_scene: Number of views per scene
        output_dir: Output directory
        template_id: Template ID (None for random [0, 3], int for fixed, Tuple[int, int] for range)
        num_objects: Number of objects per scene (None for random [1, 3], int for fixed, Tuple[int, int] for range)
        num_lights: Number of lights per scene (None for random [2, 4], int for fixed, Tuple[int, int] for range)
        objaverse_mesh_paths: List of Objaverse mesh paths
        scene_center: Scene center coordinates
        scene_bounds: Scene bounding box range
        ground_level: Ground level height
        camera_distance_range: Camera distance range
        camera_fov_range: Camera FOV range
        light_distance_range: Light distance range
        light_scale_range: Light scale range
        light_emission_range: Light emission range
        examples_dir: Examples directory path
        resolution: Rendering resolution
        spp: Samples per pixel
        base_seed: Base seed
        num_workers: Number of parallel workers
        skip_rendering: Whether to skip rendering
        gpu_memory_check: Whether to check GPU memory
        
    Returns:
        Dataset metadata dictionary
    """
    if objaverse_mesh_paths is None:
        objaverse_mesh_paths = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("============ Step 1: Generating scenes in .json ==============")
    scene_configs = generate_scene_batch(
        num_scenes=num_scenes,
        output_dir=output_dir,
        template_id=template_id,
        num_objects=num_objects,
        num_views=num_views_per_scene,
        num_lights=num_lights,
        objaverse_mesh_paths=objaverse_mesh_paths,
        scene_center=scene_center,
        scene_bounds=scene_bounds,
        ground_level=ground_level,
        camera_distance_range=camera_distance_range,
        camera_fov_range=camera_fov_range,
        light_distance_range=light_distance_range,
        light_scale_range=light_scale_range,
        light_emission_range=light_emission_range,
        examples_dir=examples_dir,
        base_seed=base_seed,
        num_workers=num_workers,
        save_json=True
    )
    
    scenes_dir = os.path.join(output_dir, "scenes")
    scene_config_paths = []
    for scene_config in scene_configs:
        scene_name = scene_config.scene_name or "scene"
        scene_config_path = os.path.join(scenes_dir, f"{scene_name}.json")
        if os.path.exists(scene_config_path):
            scene_config_paths.append(scene_config_path)
    
    print("============ Step 2: Converting scenes to HDF5 ==============")
    h5_dir = os.path.join(output_dir, "h5")
    os.makedirs(h5_dir, exist_ok=True)
    
    for scene_config_path in tqdm(scene_config_paths, desc="Converting to HDF5"):
        scene_name = os.path.splitext(os.path.basename(scene_config_path))[0]
        h5_path = os.path.join(h5_dir, f"{scene_name}.h5")
        
        if os.path.exists(h5_path):
            continue
        
        from .convert_scene import main as convert_main
        import sys
        import tempfile
        
        old_argv = sys.argv
        sys.argv = ['convert_scene.py', scene_config_path, '--output_h5_path', h5_path]
        
        try:
            convert_main()
        except SystemExit:
            pass
        except Exception as e:
            print(f"Warning: Failed to convert {scene_config_path}: {e}")
        finally:
            sys.argv = old_argv
    
    print("============ Step 3: Rendering ground truth images ============")
    render_results = render_scene_batch(
        scene_configs=scene_configs,
        scene_config_paths=scene_config_paths,
        output_dir=output_dir,
        resolution=resolution,
        spp=spp,
        save_img=True,
        dump_blend=False,
        skip_rendering=skip_rendering,
        gpu_memory_check=gpu_memory_check,
        max_concurrent=1
    )
    
    print("============ Step 4: Generating metadata ============")
    metadata = {
        'num_scenes': len(scene_configs),
        'num_views_per_scene': num_views_per_scene,
        'resolution': resolution,
        'spp': spp,
        'template_id': template_id,
        'num_objects': num_objects,
        'num_lights': num_lights,
        'scenes': []
    }
    
    for scene_config, scene_config_path in zip(scene_configs, scene_config_paths):
        scene_name = os.path.splitext(os.path.basename(scene_config_path))[0]
        h5_path = os.path.join(h5_dir, f"{scene_name}.h5")
        
        scene_metadata = {
            'scene_name': scene_name,
            'json_path': scene_config_path,
            'h5_path': h5_path if os.path.exists(h5_path) else None,
            'renders': render_results.get(scene_name, [])
        }
        
        metadata['scenes'].append(scene_metadata)
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generation complete! Output directory: {output_dir}")
    print(f"Total scenes: {len(scene_configs)}")
    print(f"Total views: {len(scene_configs) * num_views_per_scene}")
    
    return metadata
