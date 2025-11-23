"""Batch scene generator.

Generate multiple scenes in parallel.
"""
import os
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import asdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .scene_config import SceneConfig
from .scene_generator import generate_random_scene

def generate_single_scene_wrapper(args):
    """Wrapper function for single scene generation (for multiprocessing)."""
    try:
        scene_config = generate_random_scene(**args)
        return scene_config
    except Exception as e:
        print(f"Error generating scene: {e}")
        return None

def generate_scene_batch(
    num_scenes: int,
    output_dir: str,
    template_id: Optional[Union[int, Tuple[int, int]]],
    num_objects: Optional[Union[int, Tuple[int, int]]],
    num_views: int,
    num_lights: Optional[Union[int, Tuple[int, int]]],
    objaverse_mesh_paths: List[str],
    scene_center: tuple = (0.0, 0.0, 0.0),
    scene_bounds: tuple = (-0.5, 0.5),
    ground_level: float = -0.5,
    camera_distance_range: tuple = (1.5, 2.0),
    camera_fov_range: tuple = (30.0, 60.0),
    light_distance_range: tuple = (2.1, 2.7),
    light_scale_range: tuple = (2.0, 2.5),
    light_emission_range: tuple = (2500.0, 5000.0),
    examples_dir: str = "examples",
    base_seed: Optional[int] = None,
    num_workers: Optional[int] = None,
    save_json: bool = True
) -> List[SceneConfig]:
    """Generate multiple scenes in batch.
    
    Args:
        num_scenes: Number of scenes to generate
        output_dir: Output directory
        template_id: Template ID (None for random [0, 3], int for fixed value, Tuple[int, int] for range)
        num_objects: Number of objects per scene (None for random [1, 3], int for fixed, Tuple[int, int] for range)
        num_views: Number of views per scene
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
        base_seed: Base seed
        num_workers: Number of parallel workers (None for CPU cores)
        save_json: Whether to save as JSON files
        
    Returns:
        List of generated SceneConfig objects
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    scene_args = []
    for i in range(num_scenes):
        seed = base_seed + i if base_seed is not None else None
        
        if seed is not None:
            random.seed(seed)
        
        # Handle template_id: None -> [0, 3], int -> fixed, Tuple -> range
        if template_id is None:
            scene_template_id = random.randint(0, 3)
        elif isinstance(template_id, tuple):
            scene_template_id = random.randint(template_id[0], template_id[1])
        else:
            scene_template_id = template_id
        
        # Handle num_objects: None -> [1, 3], int -> fixed, Tuple -> range
        if num_objects is None:
            scene_num_objects = random.choices([1, 2, 3], weights=[50, 30, 20])[0]
        elif isinstance(num_objects, tuple):
            scene_num_objects = random.randint(num_objects[0], num_objects[1])
        else:
            scene_num_objects = num_objects
        
        # Handle num_lights: None -> [2, 4], int -> fixed, Tuple -> range
        if num_lights is None:
            scene_num_lights = random.randint(2, 4)
        elif isinstance(num_lights, tuple):
            scene_num_lights = random.randint(num_lights[0], num_lights[1])
        else:
            scene_num_lights = num_lights
        
        args = {
            'template_id': scene_template_id,
            'num_objects': scene_num_objects,
            'num_views': num_views,
            'num_lights': scene_num_lights,
            'objaverse_mesh_paths': objaverse_mesh_paths,
            'scene_center': scene_center,
            'scene_bounds': scene_bounds,
            'ground_level': ground_level,
            'camera_distance_range': camera_distance_range,
            'camera_fov_range': camera_fov_range,
            'light_distance_range': light_distance_range,
            'light_scale_range': light_scale_range,
            'light_emission_range': light_emission_range,
            'examples_dir': examples_dir,
            'seed': seed
        }
        scene_args.append(args)
    
    scene_configs = []
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_scene_wrapper, scene_args),
                total=num_scenes,
                desc="Generating scenes"
            ))
        
        scene_configs = [r for r in results if r is not None]
    else:
        for args in tqdm(scene_args, desc="Generating scenes"):
            scene_config = generate_single_scene_wrapper(args)
            if scene_config is not None:
                scene_configs.append(scene_config)
    
    if save_json:
        scenes_dir = os.path.join(output_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)
        
        for i, scene_config in enumerate(scene_configs):
            scene_name = scene_config.scene_name or f"scene_{i:04d}"
            json_path = os.path.join(scenes_dir, f"{scene_name}.json")
            
            scene_dict = asdict(scene_config)
            
            with open(json_path, 'w') as f:
                json.dump(scene_dict, f, indent=2)
    
    print(f"Generated {len(scene_configs)} scenes in {output_dir}")
    
    return scene_configs
