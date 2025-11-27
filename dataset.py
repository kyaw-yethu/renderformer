import glob
import json
import imageio
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from typing import Optional


class RenderFormerInferenceDataset(Dataset):
    """
    Base dataset class for loading RenderFormer H5 files.
    Each H5 file contains one scene with geometry and multiple camera views.
    """
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None):
        """
        Args:
            h5_folder_path: Directory containing H5 files
            padding_length: Optional padding length for triangles (for batching scenes with different triangle counts)
            load_gt: Whether to load ground truth images (for training)
        """
        self.file_list = glob.glob(os.path.join(h5_folder_path, '*.h5'))
        self.file_list = natsorted(self.file_list)
        print(f'Found {len(self.file_list)} h5 files in {h5_folder_path}')
        self.padding_length = padding_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        
        with h5py.File(file_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            num_tris = triangles.shape[0]
            texture = torch.from_numpy(np.array(f['texture'])).float()
            vn = torch.from_numpy(np.array(f['vn'])).float()
            c2w = torch.from_numpy(np.array(f['c2w'])).float()
            fov = torch.from_numpy(np.array(f['fov'])).float()

            # Optional padding for batching
            if self.padding_length is not None:
                triangles = torch.concatenate((triangles, torch.zeros(
                    (self.padding_length - num_tris, *triangles.shape[1:]))), dim=0)
                texture = torch.concatenate((texture, torch.zeros(
                    (self.padding_length - num_tris, *texture.shape[1:]))), dim=0)
                vn = torch.concatenate((vn, torch.zeros(
                    (self.padding_length - num_tris, *vn.shape[1:]))), dim=0)
                mask = torch.zeros(self.padding_length, dtype=torch.bool)
                mask[:num_tris] = True
            else:
                mask = torch.ones(num_tris, dtype=torch.bool)

            data = {
                'triangles': triangles,
                'texture': texture,
                'mask': mask,
                'vn': vn,
                'c2w': c2w,
                'fov': fov,
                'file_path': file_path
            }
            
        return data

class RenderFormerTrainingDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        split: str,
        train_ratio: float = 0.8,
        padding_length: Optional[int] = None,
    ):
        """
        Args:
            metadata_path: Path to metadata JSON file
            padding_length: Optional padding length for triangles
            split: 'train' or 'val'
            train_ratio: Ratio of training samples (default: 0.8 = 80% train, 20% val)
        """
        self.metadata_path = metadata_path
        self.padding_length = padding_length
        self.split = split
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        all_samples = []
        dataset_dir = os.path.dirname(metadata_path)
        
        # Collect all samples
        for scene_info in self.metadata['scenes']:
            scene_name = scene_info['scene_name']
            h5_path = scene_info.get('h5_path')
            
            if h5_path is None or not os.path.exists(h5_path):
                continue
            
            renders = scene_info.get('renders', [])
            
            # Collect all render paths for this scene
            exr_paths = []
            for render_info in renders:
                exr_path = render_info[0] if isinstance(render_info, (list, tuple)) else None
                if exr_path and os.path.exists(exr_path):
                    exr_paths.append(exr_path)
            
            # Only add sample if at least one render exists
            if exr_paths:
                all_samples.append({
                    'scene_name': scene_name,
                    'h5_path': h5_path,
                    'exr_paths': exr_paths
                })
        
        # Split into train and val
        num_samples = len(all_samples)
        num_train = int(num_samples * train_ratio)
        
        if split == 'train':
            self.samples = all_samples[:num_train]
            print(f'Loaded {len(self.samples)} training scenes from {metadata_path}')
        elif split == 'val':
            self.samples = all_samples[num_train:]
            print(f'Loaded {len(self.samples)} validation scenes from {metadata_path}')
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['h5_path']
        
        with h5py.File(file_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            num_tris = triangles.shape[0]
            texture = torch.from_numpy(np.array(f['texture'])).float()
            vn = torch.from_numpy(np.array(f['vn'])).float()
            
            # Load ALL camera views
            c2w = torch.from_numpy(np.array(f['c2w'])).float()  # [N_views, 4, 4]
            fov = torch.from_numpy(np.array(f['fov'])).float()  # [N_views]
        
        # Optional padding for batching
        if self.padding_length is not None:
            triangles = torch.concatenate((triangles, torch.zeros(
                (self.padding_length - num_tris, *triangles.shape[1:]))), dim=0)
            texture = torch.concatenate((texture, torch.zeros(
                (self.padding_length - num_tris, *texture.shape[1:]))), dim=0)
            vn = torch.concatenate((vn, torch.zeros(
                (self.padding_length - num_tris, *vn.shape[1:]))), dim=0)
            mask = torch.zeros(self.padding_length, dtype=torch.bool)
            mask[:num_tris] = True
        else:
            mask = torch.ones(num_tris, dtype=torch.bool)
        
        # Load ALL ground truth images
        gt_images = []
        for exr_path in sample['exr_paths']:
            gt_image = imageio.v3.imread(exr_path)
            
            if isinstance(gt_image, np.ndarray):
                gt_image = torch.from_numpy(gt_image).float()
            
            # Remove alpha channel if present
            if gt_image.shape[-1] == 4:
                gt_image = gt_image[..., :3]
            
            gt_images.append(gt_image)
        
        # Stack all images: [N_views, H, W, 3]
        gt_images = torch.stack(gt_images, dim=0)
        
        return {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,  # [N_views, 4, 4]
            'fov': fov,  # [N_views]
            'gt_images': gt_images,  # [N_views, H, W, 3]
            'file_path': file_path,
            'scene_name': sample['scene_name'],
            'num_views': len(sample['exr_paths'])
        }

# class RenderFormerTrainingDataset(Dataset):
#     def __init__(
#         self,
#         metadata_path: str,
#         padding_length: Optional[int] = None,
#     ):
#         self.metadata_path = metadata_path
#         self.padding_length = padding_length
        
#         with open(metadata_path, 'r') as f:
#             self.metadata = json.load(f)
        
#         self.samples = []
#         dataset_dir = os.path.dirname(metadata_path)
        
#         for scene_info in self.metadata['scenes']:
#             scene_name = scene_info['scene_name']
#             h5_path = scene_info.get('h5_path')
            
#             renders = scene_info.get('renders', [])
            
#             for view_idx, render_info in enumerate(renders):
#                 exr_path = render_info[0] if isinstance(render_info, (list, tuple)) else None
                
#                 if os.path.exists(exr_path):
#                     self.samples.append({
#                         'scene_name': scene_name,
#                         'view_idx': view_idx,
#                         'h5_path': h5_path,
#                         'exr_path': exr_path
#                     })
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         file_path = sample['h5_path']
#         with h5py.File(file_path, 'r') as f:
#             triangles = torch.from_numpy(np.array(f['triangles'])).float()
#             num_tris = triangles.shape[0]
#             texture = torch.from_numpy(np.array(f['texture'])).float()
#             vn = torch.from_numpy(np.array(f['vn'])).float()
            
#             view_idx = sample['view_idx']
#             c2w = torch.from_numpy(np.array(f['c2w'][view_idx:view_idx+1])).float()
#             fov = torch.from_numpy(np.array(f['fov'][view_idx:view_idx+1])).float()
        
#              # Optional padding for batching
#             if self.padding_length is not None:
#                 triangles = torch.concatenate((triangles, torch.zeros(
#                     (self.padding_length - num_tris, *triangles.shape[1:]))), dim=0)
#                 texture = torch.concatenate((texture, torch.zeros(
#                     (self.padding_length - num_tris, *texture.shape[1:]))), dim=0)
#                 vn = torch.concatenate((vn, torch.zeros(
#                     (self.padding_length - num_tris, *vn.shape[1:]))), dim=0)
#                 mask = torch.zeros(self.padding_length, dtype=torch.bool)
#                 mask[:num_tris] = True
#             else:
#                 mask = torch.ones(num_tris, dtype=torch.bool)
        
#         gt_image = imageio.v3.imread(sample['exr_path'])
        
#         if isinstance(gt_image, np.ndarray):
#             gt_image = torch.from_numpy(gt_image).float()
#         else:
#             pass
        
#         if gt_image.shape[-1] == 4:
#             gt_image = gt_image[..., :3]
        
#         return {
#             'triangles': triangles,
#             'texture': texture,
#             'mask': mask,
#             'vn': vn,
#             'c2w': c2w,
#             'fov': fov,
#             'file_path': file_path,
#             'gt_image': gt_image,
#             'scene_name': sample['scene_name'],
#             'view_idx': sample['view_idx']
#         }