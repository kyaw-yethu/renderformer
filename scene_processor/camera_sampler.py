import random
import numpy as np
from typing import List, Optional, Tuple
from .scene_config import CameraConfig

def sample_camera_position(
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance_range: Tuple[float, float] = (1.5, 2.0),
    scene_bounds: Tuple[float, float] = (-0.5, 0.5),
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Sample camera position.
    
    Args:
        scene_center:
        distance_range:
        scene_bounds:
        seed:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    distance = np.random.uniform(distance_range[0], distance_range[1])
    
    # theta: azimuth (0 to 2*pi)
    theta = np.random.uniform(0, 2 * np.pi)

    phi = np.random.uniform(0, np.pi / 2)
    
    x = distance * np.sin(phi) * np.cos(theta)
    y = distance * np.sin(phi) * np.sin(theta)
    z = distance * np.cos(phi)
    
    position = np.array(scene_center) + np.array([x, y, z])
    
    return tuple(position)

def sample_look_at(
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_scale: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """Sample look at point (scene center + noise).
    
    Args:
        scene_center:
        noise_scale:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    noise = np.random.normal(0, noise_scale, 3)
    look_at = np.array(scene_center) + noise
    
    return tuple(look_at)

def sample_fov(
    fov_range: Tuple[float, float] = (30.0, 60.0),
    seed: Optional[int] = None
) -> float:
    """Sample field of view (FOV).
    
    Args:
        fov_range:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    fov = np.random.uniform(fov_range[0], fov_range[1])
    return float(fov)

def sample_camera_up(seed: Optional[int] = None) -> Tuple[float, float, float]:
    """Sample camera up vector.
    
    Args:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    up = np.array([0.0, 0.0, 1.0])
    
    noise = np.random.normal(0, 0.05, 3)
    up = up + noise
    up = up / np.linalg.norm(up)
    
    return tuple(up)

def sample_cameras(
    num_views: int,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance_range: Tuple[float, float] = (1.5, 2.0),
    fov_range: Tuple[float, float] = (30.0, 60.0),
    scene_bounds: Tuple[float, float] = (-0.5, 0.5),
    seed: Optional[int] = None
) -> List[CameraConfig]:
    """Sample multiple cameras.
    
    Args:
        num_views:
        scene_center:
        distance_range:
        fov_range:
        scene_bounds:
        seed:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    cameras = []
    
    for i in range(num_views):

        cam_seed = seed + i if seed is not None else None
        
        position = sample_camera_position(
            scene_center=scene_center,
            distance_range=distance_range,
            scene_bounds=scene_bounds,
            seed=cam_seed
        )
        
        look_at = sample_look_at(
            scene_center=scene_center,
            noise_scale=0.1,
            seed=cam_seed
        )
        
        up = sample_camera_up(seed=cam_seed)
        
        fov = sample_fov(
            fov_range=fov_range,
            seed=cam_seed
        )
        
        camera = CameraConfig(
            position=list(position),
            look_at=list(look_at),
            up=list(up),
            fov=fov
        )
        
        cameras.append(camera)
    
    return cameras
