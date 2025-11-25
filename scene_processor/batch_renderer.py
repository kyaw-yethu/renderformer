import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from .scene_config import SceneConfig
from .scene_mesh import generate_scene_mesh
from .render_worker import wait_for_gpu_memory, cleanup_blender_temp_files
from .to_blend import scene_to_img

def render_single_scene(
    scene_config: SceneConfig,
    scene_config_path: str,
    output_dir: str,
    resolution: int = 256,
    spp: int = 512,
    save_img: bool = True,
    dump_blend: bool = False,
    skip_rendering: bool = False
) -> List[Tuple[str, str]]:
    """Render single scene.
    
    Args:
        scene_config:
        scene_config_path:
        output_dir:
        resolution:
        spp: Samples per pixel
        save_img:
        dump_blend:
        skip_rendering:
        
    Returns:
    """
    scene_name = os.path.splitext(os.path.basename(scene_config_path))[0]
    
    scene_config_dir = os.path.dirname(scene_config_path)
    mesh_path = os.path.join(output_dir, "meshes", f"{scene_name}.obj")
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    
    generate_scene_mesh(scene_config, mesh_path, scene_config_dir)
    
    output_image_base = os.path.join(output_dir, "renders", scene_name)
    os.makedirs(os.path.dirname(output_image_base), exist_ok=True)
    
    if skip_rendering:

        num_cameras = len(scene_config.cameras)
        results = [(None, None) for _ in range(num_cameras)]
    else:
        try:
            results = scene_to_img(
                scene_config=scene_config,
                mesh_path=mesh_path,
                output_image_path=output_image_base,
                save_img=save_img,
                resolution=resolution,
                spp=spp,
                dump_blend_file=dump_blend,
                skip_rendering=False
            )
        except ImportError as e:
            if 'bpy' in str(e) or 'bpy_helper' in str(e):
                print(f"Warning: Blender modules not available. Skipping GT rendering for {scene_name}.")
                print("Install with: pip install bpy==4.0.0 bpy_helper==0.0.8 --extra-index-url https://download.blender.org/pypi/")
                num_cameras = len(scene_config.cameras)
                results = [(None, None) for _ in range(num_cameras)]
            else:
                raise
    
    output_paths = []
    for i, result in enumerate(results):
        if result[0] is None:
            exr_path = None
            png_path = None
        else:
            exr_path = f"{output_image_base}_{i}.exr"
            png_path = f"{output_image_base}_{i}.png" if save_img else None
        output_paths.append((exr_path, png_path))
    
    return output_paths

def render_scene_batch(
    scene_configs: List[SceneConfig],
    scene_config_paths: List[str],
    output_dir: str,
    resolution: int = 256,
    spp: int = 512,
    save_img: bool = True,
    dump_blend: bool = False,
    skip_rendering: bool = False,
    gpu_memory_check: bool = True,
    max_concurrent: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """Render multiple scenes in batch.
    
    Args:
        scene_configs:
        scene_config_paths:
        output_dir:
        resolution:
        spp: Samples per pixel
        save_img:
        dump_blend:
        skip_rendering:
        gpu_memory_check:
        max_concurrent:
        
    Returns:
        {scene_name:
    """
    os.makedirs(output_dir, exist_ok=True)
    
    render_results = {}
    
    for scene_config, scene_config_path in tqdm(
        zip(scene_configs, scene_config_paths),
        total=len(scene_configs),
        desc="Rendering scenes"
    ):
        scene_name = os.path.splitext(os.path.basename(scene_config_path))[0]
        
        if skip_rendering:
            render_results[scene_name] = []
            continue

        if gpu_memory_check:
            wait_for_gpu_memory(target_free_gb=2.0, max_wait_seconds=60)
        
        try:
            output_paths = render_single_scene(
                scene_config=scene_config,
                scene_config_path=scene_config_path,
                output_dir=output_dir,
                resolution=resolution,
                spp=spp,
                save_img=save_img,
                dump_blend=dump_blend,
                skip_rendering=skip_rendering
            )
            
            render_results[scene_name] = output_paths
            
            cleanup_blender_temp_files()
        
        except Exception as e:
            print(f"Error rendering scene {scene_name}: {e}")
            continue
    
    return render_results
