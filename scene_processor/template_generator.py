import random
import numpy as np
from typing import Dict, Optional
from .scene_config import SceneConfig, ObjectConfig, TransformConfig, MaterialConfig
from .template_scenes import get_template_scene, TemplateScene

def generate_template_scene(
    template_id: int,
    seed: Optional[int] = None,
    examples_dir: str = "examples"
) -> SceneConfig:
    """Generate template scene.
    
    Args:
        template_id:
        seed:
        examples_dir:
        
    Returns:
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    template = get_template_scene(template_id)
    
    objects: Dict[str, ObjectConfig] = {}
    
    for i, bg_obj in enumerate(template.background_objects):
        obj_key = f"background_{i}"
        
        transform = TransformConfig(
            translation=bg_obj.translation,
            rotation=bg_obj.rotation,
            scale=bg_obj.scale,
            normalize=bg_obj.normalize
        )

        material = MaterialConfig(
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.0, 0.0, 0.0],
            roughness=0.99,
            emissive=[0.0, 0.0, 0.0],
            smooth_shading=True,
            rand_tri_diffuse_seed=None,
            random_diffuse_max=0.4,
            random_diffuse_type="per-shading-group"
        )
        
        mesh_path = f"{examples_dir}/{bg_obj.mesh_path}"
        
        objects[obj_key] = ObjectConfig(
            mesh_path=mesh_path,
            transform=transform,
            material=material,
            remesh=False,
            remesh_target_face_num=2048
        )
    
    cameras = []
    
    scene_config = SceneConfig(
        scene_name=f"template_{template_id}",
        version="1.0",
        objects=objects,
        cameras=cameras
    )
    
    return scene_config

def add_template_to_scene_config(
    scene_config: SceneConfig,
    template_id: int,
    examples_dir: str = "examples"
) -> SceneConfig:
    """Add template background to existing SceneConfig.
    
    Args:
        scene_config:
        template_id:
        examples_dir:
        
    Returns:
    """
    template = get_template_scene(template_id)
    
    new_objects = scene_config.objects.copy()
    
    bg_start_idx = len([k for k in new_objects.keys() if k.startswith("background_")])
    
    for i, bg_obj in enumerate(template.background_objects):
        obj_key = f"background_{bg_start_idx + i}"
        
        material = MaterialConfig(
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.0, 0.0, 0.0],
            roughness=0.99,
            emissive=[0.0, 0.0, 0.0],
            smooth_shading=True,
            rand_tri_diffuse_seed=None,
            random_diffuse_max=0.4,
            random_diffuse_type="per-shading-group"
        )
        
        transform = TransformConfig(
            translation=bg_obj.translation,
            rotation=bg_obj.rotation,
            scale=bg_obj.scale,
            normalize=bg_obj.normalize
        )
        
        mesh_path = f"{examples_dir}/{bg_obj.mesh_path}"
        
        new_objects[obj_key] = ObjectConfig(
            mesh_path=mesh_path,
            material=material,
            transform=transform,
            remesh=False,
            remesh_target_face_num=2048
        )
    
    return SceneConfig(
        scene_name=scene_config.scene_name,
        version=scene_config.version,
        objects=new_objects,
        cameras=scene_config.cameras
    )
