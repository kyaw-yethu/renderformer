import random
import numpy as np
import trimesh
from typing import List, Optional, Tuple, Dict
from .scene_config import ObjectConfig, TransformConfig, MaterialConfig
from .objaverse_loader import load_objaverse_mesh

def sample_object_position(
    scene_bounds: Tuple[float, float] = (-0.5, 0.5),
    ground_level: float = -0.5,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Sample random position in scene.
    
    Args:
        scene_bounds:
        ground_level:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    x = np.random.uniform(scene_bounds[0], scene_bounds[1])
    y = np.random.uniform(scene_bounds[0], scene_bounds[1])
    z = np.random.uniform(ground_level, scene_bounds[1])
    
    return (x, y, z)

def sample_object_rotation(seed: Optional[int] = None) -> Tuple[float, float, float]:
    """Sample random rotation.
    
    Args:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    rx = np.random.uniform(0, 360)
    ry = np.random.uniform(0, 360)
    rz = np.random.uniform(0, 360)
    
    return (rx, ry, rz)

def check_collision(
    mesh1: trimesh.Trimesh,
    pos1: Tuple[float, float, float],
    mesh2: trimesh.Trimesh,
    pos2: Tuple[float, float, float],
    threshold: float = 0.1
) -> bool:
    """Check collision between two meshes.
    
    Args:
        mesh1:
        pos1:
        mesh2:
        pos2:
        threshold:
        
    Returns:
    """

    bbox1 = mesh1.bounds
    bbox1_min = bbox1[0] + np.array(pos1)
    bbox1_max = bbox1[1] + np.array(pos1)
    
    bbox2 = mesh2.bounds
    bbox2_min = bbox2[0] + np.array(pos2)
    bbox2_max = bbox2[1] + np.array(pos2)
    
    collision = (
        bbox1_max[0] + threshold > bbox2_min[0] and
        bbox1_min[0] - threshold < bbox2_max[0] and
        bbox1_max[1] + threshold > bbox2_min[1] and
        bbox1_min[1] - threshold < bbox2_max[1] and
        bbox1_max[2] + threshold > bbox2_min[2] and
        bbox1_min[2] - threshold < bbox2_max[2]
    )
    
    return collision

def place_object_in_scene(
    mesh_path: str,
    scene_bounds: Tuple[float, float] = (-0.5, 0.5),
    ground_level: float = -0.5,
    existing_objects: Optional[List[Tuple[trimesh.Trimesh, Tuple[float, float, float]]]] = None,
    max_attempts: int = 50,
    seed: Optional[int] = None
) -> Optional[Tuple[ObjectConfig, Tuple[float, float, float]]]:
    """Place object in scene.
    
    Args:
        mesh_path:
        scene_bounds:
        ground_level:
        existing_objects:
        max_attempts:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    mesh = load_objaverse_mesh(mesh_path, validate=True, simplify=False)
    if mesh is None:
        return None
    
    mesh_center = mesh.vertices.mean(axis=0)
    mesh.vertices = mesh.vertices - mesh_center
    mesh_size = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max()
    if mesh_size > 0:
        mesh.vertices = mesh.vertices / mesh_size
    
    for attempt in range(max_attempts):

        position = sample_object_position(scene_bounds, ground_level, seed=None)
        
        rotation = sample_object_rotation(seed=None)
        
        scaled_coord = np.random.uniform(0.7, 1.2)
        scale = (scaled_coord, scaled_coord, scaled_coord)

        if existing_objects is not None:
            collision = False
            for existing_mesh, existing_pos in existing_objects:
                if check_collision(mesh, position, existing_mesh, existing_pos):
                    collision = True
                    break
            
            if collision:
                continue
        
        transform = TransformConfig(
            translation=list(position),
            rotation=list(rotation),
            scale=list(scale),
            normalize=True
        )
        
        material = MaterialConfig(
            diffuse=[0.5, 0.5, 0.5],
            specular=[0.0, 0.0, 0.0],
            roughness=0.5,
            emissive=[0.0, 0.0, 0.0],
            smooth_shading=True,
            rand_tri_diffuse_seed=None,
            random_diffuse_max=0.5,
            random_diffuse_type="per-shading-group"
        )
        
        obj_config = ObjectConfig(
            mesh_path=mesh_path,
            transform=transform,
            material=material,
            remesh=False,
            remesh_target_face_num=1024
        )
        
        return (obj_config, position)
    
    return None

def sample_objects_from_objaverse(
    mesh_paths: List[str],
    num_objects: int,
    scene_bounds: Tuple[float, float] = (-0.5, 0.5),
    ground_level: float = -0.5,
    seed: Optional[int] = None
) -> List[ObjectConfig]:
    """Sample objects from Objaverse and place in scene.
    
    Args:
        mesh_paths:
        num_objects:
        scene_bounds:
        ground_level:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if len(mesh_paths) == 0:
        return []
    
    # Use random.sample to avoid duplicates, or random.choices if we need more objects than available
    if num_objects <= len(mesh_paths):
        selected_paths = random.sample(mesh_paths, k=num_objects)
    else:
        # If we need more objects than available, use choices but shuffle to maximize diversity
        selected_paths = random.choices(mesh_paths, k=num_objects)
        # Shuffle to avoid clustering of same objects
        random.shuffle(selected_paths)
    
    placed_objects = []
    existing_objects = []
    
    for i, mesh_path in enumerate(selected_paths):

        obj_seed = seed + i if seed is not None else None
        
        result = place_object_in_scene(
            mesh_path=mesh_path,
            scene_bounds=scene_bounds,
            ground_level=ground_level,
            existing_objects=existing_objects,
            max_attempts=50,
            seed=obj_seed
        )
        
        if result is not None:
            obj_config, position = result
            placed_objects.append(obj_config)
            
            mesh = load_objaverse_mesh(mesh_path, validate=False, simplify=False)
            if mesh is not None:

                mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
                mesh_size = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max()
                if mesh_size > 0:
                    mesh.vertices = mesh.vertices / mesh_size
                mesh.apply_scale(obj_config.transform.scale)
                mesh.apply_translation(position)
                existing_objects.append((mesh, position))
    
    return placed_objects
