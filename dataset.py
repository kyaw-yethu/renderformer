import glob
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from natsort import natsorted
from typing import Optional


class RenderFormerH5Dataset(Dataset):
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
            
            # Load ground truth images if needed (for training)
            if self.load_gt:
                if 'gt_images' in f:
                    gt_images = torch.from_numpy(np.array(f['gt_images'])).float()
                    data['gt_images'] = gt_images
                else:
                    raise KeyError(f"Ground truth images not found in {file_path}. Cannot use for training.")
        
        return data


# Convenience aliases for clarity
class RenderFormerInferenceDataset(RenderFormerH5Dataset):
    """Dataset for inference (no ground truth needed)"""
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None):
        super().__init__(h5_folder_path, padding_length, load_gt=False)


class RenderFormerTrainingDataset(RenderFormerH5Dataset):
    """Dataset for training (requires ground truth images)"""
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None):
        super().__init__(h5_folder_path, padding_length, load_gt=True)