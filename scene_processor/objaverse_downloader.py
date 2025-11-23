import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
import trimesh
import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not found. Install with: pip install datasets")

def get_objaverse_object_id_hash(object_id: str) -> str:
    """객체 ID의 해시값을 계산합니다 (캐시 키로 사용)."""
    return hashlib.md5(object_id.encode()).hexdigest()

def download_objaverse_objects(
    object_ids: List[str],
    output_dir: str,
    cache_dir: Optional[str] = None,
    convert_to_obj: bool = True,
    max_objects: Optional[int] = None
) -> Dict[str, str]:
    """Objaverse-XL에서 객체를 다운로드합니다.
    
    Args:
        object_ids:
        output_dir:
        cache_dir:
        convert_to_obj:
        max_objects:
        
    Returns:
        {object_id:
    """
    if not HAS_DATASETS:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets"
        )
    
    if cache_dir is None:
        cache_dir = os.path.join(output_dir, "objaverse_cache")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    if max_objects is not None:
        object_ids = object_ids[:max_objects]
    
    downloaded_paths = {}
    
    print(f"Downloading {len(object_ids)} objects from Objaverse-XL...")
    
    try:

        dataset = load_dataset("allenai/objaverse-xl", cache_dir=cache_dir)
        
        if 'train' in dataset:
            data_split = dataset['train']
        else:
            data_split = dataset[list(dataset.keys())[0]]
        
        for obj_id in object_ids:
            try:

                obj_data = None
                for item in data_split:
                    if item.get('uid') == obj_id or item.get('id') == obj_id:
                        obj_data = item
                        break
                
                if obj_data is None:
                    print(f"Warning: Object {obj_id} not found in dataset")
                    continue
                
                glb_path = None
                if 'glb_path' in obj_data:
                    glb_path = obj_data['glb_path']
                elif 'file_path' in obj_data:
                    glb_path = obj_data['file_path']
                elif 'path' in obj_data:
                    glb_path = obj_data['path']
                
                if glb_path is None:
                    print(f"Warning: No mesh path found for {obj_id}")
                    continue
                
                if os.path.exists(glb_path):
                    mesh_path = glb_path
                else:

                    cache_path = os.path.join(cache_dir, get_objaverse_object_id_hash(obj_id))
                    if os.path.exists(cache_path):
                        mesh_path = cache_path
                    else:
                        print(f"Warning: File not found for {obj_id}: {glb_path}")
                        continue
                
                if convert_to_obj:
                    output_path = os.path.join(output_dir, f"{obj_id}.obj")
                    if not os.path.exists(output_path):
                        try:
                            mesh = trimesh.load(mesh_path, process=False)
                            if isinstance(mesh, trimesh.Scene):

                                mesh = list(mesh.geometry.values())[0]
                            
                            if isinstance(mesh, trimesh.Trimesh):
                                mesh.export(output_path)
                                print(f"Converted {obj_id} to {output_path}")
                            else:
                                print(f"Warning: {obj_id} is not a valid mesh")
                                continue
                        except Exception as e:
                            print(f"Error converting {obj_id}: {e}")
                            continue
                    else:
                        print(f"Using cached {output_path}")
                else:
                    output_path = mesh_path
                
                downloaded_paths[obj_id] = output_path
                
            except Exception as e:
                print(f"Error processing {obj_id}: {e}")
                continue
    
    except Exception as e:
        print(f"Error loading Objaverse-XL dataset: {e}")
        print("Falling back to manual download method...")

        raise NotImplementedError("Manual download method not yet implemented")
    
    print(f"Successfully downloaded {len(downloaded_paths)} objects")
    return downloaded_paths

def download_objaverse_objects_simple(
    object_ids: List[str],
    output_dir: str,
    objaverse_root: Optional[str] = None
) -> Dict[str, str]:
    """Objaverse 객체를 로컬 경로에서 로드합니다 (이미 다운로드된 경우).
    
    Args:
        object_ids:
        output_dir:
        objaverse_root:
        
    Returns:
        {object_id:
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_paths = {}
    
    if objaverse_root is None:
        objaverse_root = output_dir
    
    print(f"Loading {len(object_ids)} objects from {objaverse_root}...")
    
    for obj_id in object_ids:

        extensions = ['.glb', '.gltf', '.obj']
        found = False
        
        for ext in extensions:

            possible_paths = [
                os.path.join(objaverse_root, f"{obj_id}{ext}"),
                os.path.join(objaverse_root, obj_id, f"model{ext}"),
                os.path.join(objaverse_root, obj_id[:2], obj_id, f"model{ext}"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):

                    if ext != '.obj':
                        output_path = os.path.join(output_dir, f"{obj_id}.obj")
                        if not os.path.exists(output_path):
                            try:
                                mesh = trimesh.load(path, process=False)
                                if isinstance(mesh, trimesh.Scene):
                                    mesh = list(mesh.geometry.values())[0]
                                
                                if isinstance(mesh, trimesh.Trimesh):
                                    mesh.export(output_path)
                                    print(f"Converted {obj_id} to {output_path}")
                                else:
                                    continue
                            except Exception as e:
                                print(f"Error converting {obj_id}: {e}")
                                continue
                        else:
                            output_path = os.path.join(output_dir, f"{obj_id}.obj")
                    else:
                        output_path = path
                    
                    downloaded_paths[obj_id] = output_path
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            print(f"Warning: Object {obj_id} not found")
    
    print(f"Successfully loaded {len(downloaded_paths)} objects")
    return downloaded_paths
