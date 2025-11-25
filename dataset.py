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
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None, load_gt: bool = False):
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
        self.load_gt = load_gt

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
        padding_length: Optional[int] = None,
    ):
        self.metadata_path = metadata_path
        self.padding_length = padding_length
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = []
        dataset_dir = os.path.dirname(metadata_path)
        
        for scene_info in self.metadata['scenes']:
            scene_name = scene_info['scene_name']
            h5_path = scene_info.get('h5_path')
            
            renders = scene_info.get('renders', [])
            
            for view_idx, render_info in enumerate(renders):
                exr_path = render_info[0] if isinstance(render_info, (list, tuple)) else None
                
                if os.path.exists(exr_path):
                    self.samples.append({
                        'scene_name': scene_name,
                        'view_idx': view_idx,
                        'h5_path': h5_path,
                        'exr_path': exr_path
                    })
    
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
            
            view_idx = sample['view_idx']
            c2w = torch.from_numpy(np.array(f['c2w'][view_idx])).float()
            fov = torch.from_numpy(np.array(f['fov'][view_idx])).float()
        
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
        
        gt_image = imageio.v3.imread(sample['exr_path'])
        
        if isinstance(gt_image, np.ndarray):
            gt_image = torch.from_numpy(gt_image).float()
        else:
            pass
        
        if gt_image.shape[-1] == 4:
            gt_image = gt_image[..., :3]
        
        return {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,
            'fov': fov,
            'file_path': file_path,
            'gt_image': gt_image,
            'scene_name': sample['scene_name'],
            'view_idx': sample['view_idx']
        }