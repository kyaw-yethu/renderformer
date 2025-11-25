import os
import trimesh
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path

def load_objaverse_mesh(
    mesh_path: str,
    validate: bool = True,
    simplify: bool = False,
    max_faces: Optional[int] = None
) -> Optional[trimesh.Trimesh]:
    """Objaverse 메시를 로드하고 전처리합니다.
    
    Args:
        mesh_path:
        validate:
        simplify:
        max_faces:
        
    Returns:
    """
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file not found: {mesh_path}")
        return None
    
    try:
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if len(geometries) == 0:
                print(f"Warning: Empty scene: {mesh_path}")
                return None
            mesh = geometries[0]
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Warning: Not a valid mesh: {mesh_path}")
            return None
        
        if validate:
            if not mesh.is_volume:

                pass
            
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"Warning: Empty mesh: {mesh_path}")
                return None
            
            if np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)):
                print(f"Warning: Invalid vertices in {mesh_path}")
                return None
        
        if simplify and max_faces is not None and len(mesh.faces) > max_faces:
            try:

                mesh = mesh.simplify_quadric_decimation(max_faces)
            except Exception as e:
                print(f"Warning: Failed to simplify {mesh_path}: {e}")
        
        return mesh
    
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None

def filter_valid_meshes(
    mesh_paths: List[str],
    min_faces: int = 10,
    max_faces: int = 100000,
    min_vertices: int = 4
) -> List[str]:
    """유효한 메시만 필터링합니다.
    
    Args:
        mesh_paths:
        min_faces:
        max_faces:
        min_vertices:
        
    Returns:
    """
    valid_paths = []
    
    for mesh_path in mesh_paths:
        mesh = load_objaverse_mesh(mesh_path, validate=True, simplify=False)
        
        if mesh is None:
            continue
        
        num_faces = len(mesh.faces)
        num_vertices = len(mesh.vertices)
        
        if num_faces < min_faces or num_faces > max_faces:
            continue
        
        if num_vertices < min_vertices:
            continue
        
        valid_paths.append(mesh_path)
    
    return valid_paths

def get_mesh_metadata(mesh_path: str) -> Optional[Dict]:
    """메시 메타데이터를 가져옵니다.
    
    Args:
        mesh_path:
        
    Returns:
    """
    mesh = load_objaverse_mesh(mesh_path, validate=False, simplify=False)
    
    if mesh is None:
        return None
    
    bbox = mesh.bounds
    bbox_size = bbox[1] - bbox[0]
    
    metadata = {
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'is_watertight': mesh.is_watertight,
        'is_winding_consistent': mesh.is_winding_consistent,
        'bounding_box': bbox.tolist(),
        'bounding_box_size': bbox_size.tolist(),
        'volume': float(mesh.volume) if mesh.is_volume else None,
        'surface_area': float(mesh.area),
    }
    
    return metadata
