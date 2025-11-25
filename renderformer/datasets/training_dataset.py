import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple
import imageio

class RenderFormerTrainingDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        padding_length: Optional[int] = None,
        use_ldr: bool = False,
        transform: Optional[callable] = None
    ):
        self.metadata_path = metadata_path
        self.padding_length = padding_length
        self.use_ldr = use_ldr
        self.transform = transform
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = []
        dataset_dir = os.path.dirname(metadata_path)
        
        for scene_info in self.metadata['scenes']:
            scene_name = scene_info['scene_name']
            h5_path = scene_info.get('h5_path')
            
            if h5_path is None or not os.path.exists(h5_path):
                continue
            
            if not os.path.isabs(h5_path):
                h5_path = os.path.join(dataset_dir, h5_path)
            
            renders = scene_info.get('renders', [])
            
            for view_idx, render_info in enumerate(renders):
                exr_path = render_info[0] if isinstance(render_info, (list, tuple)) else None
                
                if exr_path is None:

                    renders_dir = os.path.join(dataset_dir, "renders")
                    exr_path = os.path.join(renders_dir, f"{scene_name}_view_{view_idx}.exr")
                
                if not os.path.isabs(exr_path):
                    exr_path = os.path.join(dataset_dir, exr_path)
                
                if os.path.exists(exr_path):
                    self.samples.append({
                        'scene_name': scene_name,
                        'view_idx': view_idx,
                        'h5_path': h5_path,
                        'exr_path': exr_path
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        with h5py.File(sample['h5_path'], 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            num_tris = triangles.shape[0]
            texture = torch.from_numpy(np.array(f['texture'])).float()
            vn = torch.from_numpy(np.array(f['vn'])).float()
            
            view_idx = sample['view_idx']
            c2w = torch.from_numpy(np.array(f['c2w'][view_idx])).float()
            fov = torch.from_numpy(np.array(f['fov'][view_idx:view_idx+1])).float()
        
        if self.padding_length is not None:
            triangles = torch.cat((
                triangles,
                torch.zeros((self.padding_length - num_tris, *triangles.shape[1:]))
            ), dim=0)
            texture = torch.cat((
                texture,
                torch.zeros((self.padding_length - num_tris, *texture.shape[1:]))
            ), dim=0)
            vn = torch.cat((
                vn,
                torch.zeros((self.padding_length - num_tris, *vn.shape[1:]))
            ), dim=0)
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
        
        if self.use_ldr:
            gt_image = torch.clamp(gt_image, 0, 1)
            gt_image = torch.pow(gt_image, 1.0 / 2.2)  # gamma correction
        
        if self.transform is not None:
            data = {
                'triangles': triangles,
                'texture': texture,
                'vn': vn,
                'c2w': c2w,
                'fov': fov,
                'mask': mask,
                'gt_image': gt_image
            }
            data = self.transform(data)
            return data
        
        return {
            'triangles': triangles,
            'texture': texture,
            'vn': vn,
            'c2w': c2w,
            'fov': fov,
            'mask': mask,
            'gt_image': gt_image,
            'scene_name': sample['scene_name'],
            'view_idx': sample['view_idx']
        }
