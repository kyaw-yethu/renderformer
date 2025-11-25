import random
import numpy as np
from typing import Optional, List, Dict
from .scene_config import SceneConfig
from .template_generator import generate_template_scene
from .object_placer import sample_objects_from_objaverse
from .camera_sampler import sample_cameras
from .light_sampler import sample_lights
from .material_sampler import sample_materials

def generate_random_scene(
    template_id: int,
    num_objects: int,
    num_views: int,
    num_lights: int,
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
    max_triangles: int = 2048,
    seed: Optional[int] = None
) -> SceneConfig:
    """Generate random scene.
    
    Args:
        template_id:
        num_objects:
        num_views:
        num_lights:
        objaverse_mesh_paths:
        scene_center:
        scene_bounds:
        ground_level:
        camera_distance_range:
        camera_fov_range:
        light_distance_range:
        light_scale_range:
        light_emission_range:
        examples_dir:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    scene_config = generate_template_scene(
        template_id=template_id,
        seed=seed,
        examples_dir=examples_dir
    )
    
    if num_objects > 0 and len(objaverse_mesh_paths) > 0:
        obj_seed = seed + 1000 if seed is not None else None
        placed_objects = sample_objects_from_objaverse(
            mesh_paths=objaverse_mesh_paths,
            num_objects=num_objects,
            scene_bounds=scene_bounds,
            ground_level=ground_level,
            max_triangles=max_triangles,
            seed=obj_seed
        )
        
        for i, obj_config in enumerate(placed_objects):
            obj_key = f"object_{i}"
            scene_config.objects[obj_key] = obj_config
    
    if num_lights > 0:
        light_seed = seed + 2000 if seed is not None else None
        lights = sample_lights(
            num_lights=num_lights,
            scene_center=scene_center,
            distance_range=light_distance_range,
            scale_range=light_scale_range,
            total_emission_range=light_emission_range,
            examples_dir=examples_dir,
            seed=light_seed
        )
        
        for i, light_obj in enumerate(lights):
            light_key = f"light_{i}"
            scene_config.objects[light_key] = light_obj
    
    material_seed = seed + 3000 if seed is not None else None
    scene_config.objects = sample_materials(
        objects=scene_config.objects,
        seed=material_seed
    )
    
    camera_seed = seed + 4000 if seed is not None else None
    cameras = sample_cameras(
        num_views=num_views,
        template_id=template_id,
        scene_center=scene_center,
        distance_range=camera_distance_range,
        fov_range=camera_fov_range,
        scene_bounds=scene_bounds,
        seed=camera_seed
    )
    
    scene_config.cameras = cameras
    
    scene_config.scene_name = f"scene_template_{template_id}_seed_{seed}"
    
    return scene_config
