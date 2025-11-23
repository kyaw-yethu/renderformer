"""Material sampler.

per-vertex normal vs flat shading (50:50)
"""
import random
import numpy as np
from typing import List, Optional, Dict
from .scene_config import ObjectConfig, MaterialConfig

def sample_diffuse_color(
    min_value: float = 0.1,
    max_value: float = 0.9,
    seed: Optional[int] = None
) -> List[float]:
    """Sample diffuse color."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    r = np.random.uniform(min_value, max_value)
    g = np.random.uniform(min_value, max_value)
    b = np.random.uniform(min_value, max_value)
    
    return [float(r), float(g), float(b)]

def sample_specular_color(
    max_specular: float = 0.3,
    seed: Optional[int] = None
) -> List[float]:
    """Sample specular color (monochrome, white)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    specular_value = np.random.uniform(0.0, max_specular)
    
    return [float(specular_value), float(specular_value), float(specular_value)]

def sample_roughness(
    min_roughness: float = 0.01,
    max_roughness: float = 1.0,
    seed: Optional[int] = None
) -> float:
    """Sample roughness."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    roughness = np.random.uniform(min_roughness, max_roughness)
    return float(roughness)

def sample_material_assignment_type(seed: Optional[int] = None) -> str:
    """Sample material assignment type (per-object vs per-triangle)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    return "per-object" if random.random() < 0.5 else "per-triangle"

def sample_shading_type(seed: Optional[int] = None) -> bool:
    """Sample shading type (smooth vs flat)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    return random.random() < 0.5

def sample_materials(
    objects: Dict[str, ObjectConfig],
    min_diffuse: float = 0.1,
    max_diffuse: float = 0.9,
    max_specular: float = 0.3,
    min_roughness: float = 0.01,
    max_roughness: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, ObjectConfig]:
    """Sample and assign materials to objects."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    updated_objects = {}
    
    for obj_key, obj_config in objects.items():

        if obj_config.material.emissive[0] > 0:
            updated_objects[obj_key] = obj_config
            continue
        
        assignment_type = sample_material_assignment_type(seed=seed)
        
        smooth_shading = sample_shading_type(seed=seed)
        
        diffuse = sample_diffuse_color(
            min_value=min_diffuse,
            max_value=max_diffuse,
            seed=seed
        )
        
        specular = sample_specular_color(
            max_specular=max_specular,
            seed=seed
        )
        
        total = max(diffuse) + max(specular)
        if total > 1.0:

            max_allowed_specular = 1.0 - max(diffuse)
            specular = [min(s, max_allowed_specular) for s in specular]
        
        roughness = sample_roughness(
            min_roughness=min_roughness,
            max_roughness=max_roughness,
            seed=seed
        )
        
        rand_tri_diffuse_seed = None
        random_diffuse_max = max_diffuse
        
        if assignment_type == "per-triangle":
            base_seed = seed if seed is not None else 0
            rand_tri_diffuse_seed = (hash(obj_key) + base_seed) & 0xFFFFFFFF
            random_diffuse_max = max_diffuse
        
        material = MaterialConfig(
            diffuse=diffuse,
            specular=specular,
            roughness=roughness,
            emissive=[0.0, 0.0, 0.0],
            smooth_shading=smooth_shading,
            rand_tri_diffuse_seed=rand_tri_diffuse_seed,
            random_diffuse_max=random_diffuse_max,
            random_diffuse_type="per-triangle" if assignment_type == "per-triangle" else "per-shading-group"
        )
        
        updated_obj = ObjectConfig(
            mesh_path=obj_config.mesh_path,
            material=material,
            transform=obj_config.transform,
            remesh=obj_config.remesh,
            remesh_target_face_num=obj_config.remesh_target_face_num
        )
        
        updated_objects[obj_key] = updated_obj
        
        if seed is not None:
            seed += 1
    
    return updated_objects
