from typing import Literal, Optional
import torch
import numpy as np

def compute_triangle_centers(tri_vpos: torch.Tensor) -> torch.Tensor:
    vertices = tri_vpos.view(*tri_vpos.shape[:-1], 3, 3)
    centers = vertices.mean(dim=-2)
    return centers

def cluster_triangles_grid(
    tri_vpos: torch.Tensor,
    window_size: int,
    valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size, num_triangles = tri_vpos.shape[:2]
    device = tri_vpos.device
    dtype = tri_vpos.dtype
    
    centers = compute_triangle_centers(tri_vpos)  # (B, N, 3)
    
    cluster_ids = torch.zeros((batch_size, num_triangles), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        if valid_mask is not None:
            valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue
            valid_centers = centers[b, valid_indices]  # (num_valid, 3)
        else:
            valid_indices = torch.arange(num_triangles, device=device)
            valid_centers = centers[b]
        
        if len(valid_centers) == 0:
            continue
        
        min_coords = valid_centers.min(dim=0)[0]  # (3,)
        max_coords = valid_centers.max(dim=0)[0]  # (3,)
        bbox_size = max_coords - min_coords
        
        num_valid = len(valid_centers)
        num_clusters = max(1, (num_valid + window_size - 1) // window_size)
        
        if torch.all(bbox_size > 1e-6):
            grid_ratios = bbox_size / bbox_size.max()
            grid_size = (num_clusters / (grid_ratios[0] * grid_ratios[1] * grid_ratios[2])) ** (1.0 / 3.0)
            grid_x = max(1, int(grid_ratios[0] * grid_size))
            grid_y = max(1, int(grid_ratios[1] * grid_size))
            grid_z = max(1, int(grid_ratios[2] * grid_size))
            
            while grid_x * grid_y * grid_z < num_clusters:
                if grid_x <= grid_y and grid_x <= grid_z:
                    grid_x += 1
                elif grid_y <= grid_z:
                    grid_y += 1
                else:
                    grid_z += 1
        else:
            grid_x = num_clusters
            grid_y = 1
            grid_z = 1
        
        normalized_centers = (valid_centers - min_coords) / (bbox_size + 1e-8)
        
        grid_indices_x = torch.clamp((normalized_centers[:, 0] * grid_x).long(), 0, grid_x - 1)
        grid_indices_y = torch.clamp((normalized_centers[:, 1] * grid_y).long(), 0, grid_y - 1)
        grid_indices_z = torch.clamp((normalized_centers[:, 2] * grid_z).long(), 0, grid_z - 1)
        
        cluster_id_flat = grid_indices_x * (grid_y * grid_z) + grid_indices_y * grid_z + grid_indices_z
        
        unique_cluster_ids = torch.unique(cluster_id_flat)
        id_mapping = torch.zeros(unique_cluster_ids.max().item() + 1, dtype=torch.long, device=device)
        id_mapping[unique_cluster_ids] = torch.arange(len(unique_cluster_ids), device=device)
        cluster_id_flat = id_mapping[cluster_id_flat]
        
        cluster_ids[b, valid_indices] = cluster_id_flat
    
    return cluster_ids

def cluster_triangles_spatial(
    tri_vpos: torch.Tensor,
    window_size: int,
    method: Literal['grid'] = 'grid',
    valid_mask: Optional[torch.Tensor] = None,
    num_clusters: Optional[int] = None,
    random_state: int = 42
) -> torch.Tensor:
    """
    
    Args:
        tri_vpos:
        window_size:
        method:
        valid_mask:
        num_clusters:
        random_state:
    
    Returns:
        cluster_ids:
    """
    if method == 'grid':
        return cluster_triangles_grid(tri_vpos, window_size, valid_mask)
