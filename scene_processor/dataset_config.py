"""Dataset generation configuration.

Hyperparameter definitions including RTX 3090 optimized defaults.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

@dataclass
class DatasetConfig:
    """Dataset generation configuration.
    
    template_id, num_objects, num_lights can be:
    - None: Use default random range (template_id: [0, 3], num_objects: [1, 3], num_lights: [2, 4])
    - int: Fixed value
    - Tuple[int, int]: Random range [min, max] (inclusive)
    """
    
    num_scenes: int = 100
    num_views_per_scene: int = 4
    template_id: Optional[Union[int, Tuple[int, int]]] = None
    num_objects: Optional[Union[int, Tuple[int, int]]] = None
    num_lights: Optional[Union[int, Tuple[int, int]]] = None
    
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_bounds: Tuple[float, float] = (-0.5, 0.5)
    ground_level: float = -0.5
    
    camera_distance_range: Tuple[float, float] = (1.5, 2.0)
    camera_fov_range: Tuple[float, float] = (30.0, 60.0)
    
    light_distance_range: Tuple[float, float] = (2.1, 2.7)
    light_scale_range: Tuple[float, float] = (2.0, 2.5)
    light_emission_range: Tuple[float, float] = (2500.0, 5000.0)
    
    resolution: int = 256
    spp: int = 512
    max_triangles: int = 2048
    min_triangles: int = 100
    
    objaverse_mesh_paths: Optional[List[str]] = None
    objaverse_root: Optional[str] = None
    
    examples_dir: str = "examples"
    output_dir: str = "output/dataset"
    
    num_workers: Optional[int] = None
    
    skip_rendering: bool = False
    gpu_memory_check: bool = True
    
    base_seed: Optional[int] = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DatasetConfig':
        # Convert lists to tuples for template_id, num_objects, num_lights
        # JSON doesn't support tuples, so we use lists and convert them
        config_dict = config_dict.copy()
        for key in ['template_id', 'num_objects', 'num_lights']:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
    
    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DatasetConfig':
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
