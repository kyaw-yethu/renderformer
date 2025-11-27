import random
import numpy as np
from typing import List, Optional, Tuple
from .scene_config import ObjectConfig, TransformConfig, MaterialConfig

    
def sample_light_position(
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance_range: Tuple[float, float] = (2.1, 2.7),
    elevation_range: Tuple[float, float] = (1.2, 2.5),
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Sample light position.
    
    Args:
        scene_center:
        distance_range:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    distance = np.random.uniform(distance_range[0], distance_range[1])
    
    theta = np.random.uniform(0, 2 * np.pi)

    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = np.random.uniform(elevation_range[0], elevation_range[1])

    position = np.array(scene_center) + np.array([x, y, z])
    
    return tuple(position)

def sample_light_rotation(seed: Optional[int] = None) -> Tuple[float, float, float]:
    """Sample light rotation.
    
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

def sample_light_scale(
    scale_range: Tuple[float, float] = (2.0, 2.5),
    uniform: bool = True,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Sample light scale.
    
    Args:
        scale_range:
        uniform:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if uniform:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return (scale, scale, scale)
    else:
        sx = np.random.uniform(scale_range[0], scale_range[1])
        sy = np.random.uniform(scale_range[0], scale_range[1])
        sz = np.random.uniform(scale_range[0], scale_range[1])
        return (sx, sy, sz)

def sample_light_emission(
    total_emission_range: Tuple[float, float] = (2500.0, 5000.0),
    num_lights: int = 1,
    seed: Optional[int] = None
) -> List[float]:
    """Sample light emission value.
    
    Args:
        total_emission_range:
        num_lights:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    total_emission = np.random.uniform(total_emission_range[0], total_emission_range[1])
    
    if num_lights > 1:

        weights = np.random.uniform(0.5, 1.5, num_lights)
        weights = weights / weights.sum()
        emissions = [total_emission * w for w in weights]
    else:
        emissions = [total_emission]
    
    return [[e, e, e] for e in emissions]

def sample_lights(
    num_lights: int,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance_range: Tuple[float, float] = (2.1, 2.7),
    scale_range: Tuple[float, float] = (2.0, 2.5),
    total_emission_range: Tuple[float, float] = (2500.0, 5000.0),
    light_mesh_path: str = "templates/lighting/tri.obj",
    examples_dir: str = "examples",
    seed: Optional[int] = None
) -> List[ObjectConfig]:
    """Sample multiple lights.
    
    Args:
        num_lights:
        scene_center:
        distance_range:
        scale_range:
        total_emission_range:
        light_mesh_path:
        examples_dir:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    num_lights = min(num_lights, 8)
    
    emissions = sample_light_emission(
        total_emission_range=total_emission_range,
        num_lights=num_lights,
        seed=seed
    )
    
    lights = []
    
    for i in range(num_lights):

        light_seed = seed + i if seed is not None else None
        
        position = sample_light_position(
            scene_center=scene_center,
            distance_range=distance_range,
            seed=light_seed
        )
        
        rotation = sample_light_rotation(seed=light_seed)
        
        scale = sample_light_scale(
            scale_range=scale_range,
            uniform=True,
            seed=light_seed
        )
        
        material = MaterialConfig(
            diffuse=[1.0, 1.0, 1.0],
            specular=[0.0, 0.0, 0.0],
            roughness=1.0,
            emissive=emissions[i],
            smooth_shading=False,
            rand_tri_diffuse_seed=None,
            random_diffuse_max=0.0,
            random_diffuse_type="per-shading-group"
        )
        
        transform = TransformConfig(
            translation=list(position),
            rotation=list(rotation),
            scale=list(scale),
            normalize=False
        )
        
        mesh_path = f"{examples_dir}/{light_mesh_path}"
        
        light_obj = ObjectConfig(
            mesh_path=mesh_path,
            material=material,
            transform=transform,
            remesh=False,
            remesh_target_face_num=2048
        )
        
        lights.append(light_obj)
    
    return lights
