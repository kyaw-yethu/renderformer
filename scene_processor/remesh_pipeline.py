import os
import trimesh
import numpy as np
from typing import Optional, Tuple, List
from .remesh import remesh

def clean_mesh(
    mesh: trimesh.Trimesh,
    remove_internal: bool = True,
    remove_duplicate_faces: bool = True
) -> trimesh.Trimesh:
    """Clean mesh (remove internal/degenerate triangles).
    
    Args:
        mesh:
        remove_internal:
        remove_duplicate_faces:
        
    Returns:
    """

    if remove_duplicate_faces:
        mesh.remove_duplicate_faces()
    
    mesh.merge_vertices()
    
    mesh.remove_degenerate_faces()
    
    if remove_internal and mesh.is_watertight:

        pass
    
    return mesh

def remesh_to_target(
    mesh: trimesh.Trimesh,
    target_face_num: int,
    min_faces: int = 100,
    max_faces: int = 8192
) -> trimesh.Trimesh:
    """Remesh mesh to target triangle number.
    
    Args:
        mesh:
        target_face_num:
        min_faces:
        max_faces:
        
    Returns:
    """

    target_face_num = max(min_faces, min(max_faces, target_face_num))
    
    current_faces = len(mesh.faces)
    
    if min_faces <= current_faces <= max_faces:
        if abs(current_faces - target_face_num) < 50:
            return mesh
    
    try:
        vertices, faces = remesh(
            mesh.vertices,
            mesh.faces,
            target_face_num=target_face_num
        )
        
        remeshed_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )
        
        return remeshed_mesh
    
    except Exception as e:
        print(f"Warning: Remeshing failed: {e}")
        return mesh

def process_mesh_for_scene(
    mesh_path: str,
    target_face_num: int = 2048,
    min_faces: int = 100,
    max_faces: int = 4096,
    clean: bool = True
) -> Optional[trimesh.Trimesh]:
    """Preprocess mesh for scene.
    
    Args:
        mesh_path:
        target_face_num:
        min_faces:
        max_faces:
        clean:
        
    Returns:
    """
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file not found: {mesh_path}")
        return None
    
    try:

        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]
        
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Warning: Not a valid mesh: {mesh_path}")
            return None
        
        if clean:
            mesh = clean_mesh(mesh)
        
        mesh = remesh_to_target(
            mesh,
            target_face_num=target_face_num,
            min_faces=min_faces,
            max_faces=max_faces
        )
        
        return mesh
    
    except Exception as e:
        print(f"Error processing mesh {mesh_path}: {e}")
        return None

def estimate_total_triangles(
    mesh_paths: List[str],
    target_per_object: int = 512
) -> int:
    """Estimate total number of triangles for multiple meshes.
    
    Args:
        mesh_paths:
        target_per_object:
        
    Returns:
    """
    return len(mesh_paths) * target_per_object

def adjust_target_face_num_for_budget(
    current_face_num: int,
    total_budget: int,
    num_objects: int
) -> int:
    """Adjust target face number to fit triangle budget.
    
    Args:
        current_face_num:
        total_budget:
        num_objects:
        
    Returns:
    """
    if num_objects == 0:
        return current_face_num
    
    per_object_budget = total_budget // num_objects
    
    return min(current_face_num, per_object_budget)
