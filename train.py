import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import lpips

from renderformer import RenderFormerRenderingPipeline
from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer
from simple_ocio import ToneMapper
from dataset import RenderFormerTrainingDataset  # Import shared dataset

def random_rotation_augmentation(triangles, vn, c2w):
    """
    Apply random rotation to scene geometry and cameras for rotation invariance.
    Uses RoMa (Rotation Matrices) for stable rotation representation.
    
    Args:
        triangles: Triangle vertices [B, num_tris, 3, 3]
        vn: Vertex normals [B, num_tris, 3, 3]
        c2w: Camera-to-world matrices [B, N_views, 4, 4]
    
    Returns:
        Rotated triangles, vertex normals, and camera matrices
    """
    B = triangles.shape[0]
    device = triangles.device
    
    # Generate random rotation matrices for each batch element
    # Sample random axis and angle
    axes = torch.randn(B, 3, device=device)
    axes = axes / (torch.norm(axes, dim=1, keepdim=True) + 1e-8)
    angles = torch.rand(B, device=device) * 2 * np.pi
    
    # Rodrigues' rotation formula to create rotation matrix
    # R = I + sin(θ)K + (1-cos(θ))K²
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    rotation_matrices = []
    for i in range(B):
        axis = axes[i]
        cos_a = cos_angles[i]
        sin_a = sin_angles[i]
        
        # Skew-symmetric matrix K
        K = torch.zeros(3, 3, device=device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        # Rotation matrix
        I = torch.eye(3, device=device)
        R = I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)
        rotation_matrices.append(R)
    
    rotation_matrices = torch.stack(rotation_matrices)  # [B, 3, 3]
    
    # Rotate triangles
    # triangles: [B, num_tris, 3, 3] -> [B, num_tris*3, 3]
    B, num_tris, _, _ = triangles.shape
    triangles_flat = triangles.reshape(B, num_tris * 3, 3)
    triangles_rotated = torch.bmm(triangles_flat, rotation_matrices.transpose(1, 2))
    triangles_rotated = triangles_rotated.reshape(B, num_tris, 3, 3)
    
    # Rotate vertex normals
    vn_flat = vn.reshape(B, num_tris * 3, 3)
    vn_rotated = torch.bmm(vn_flat, rotation_matrices.transpose(1, 2))
    vn_rotated = vn_rotated.reshape(B, num_tris, 3, 3)
    
    # Rotate cameras
    # c2w: [B, N_views, 4, 4]
    N_views = c2w.shape[1]
    c2w_rotated = c2w.clone()
    
    # Create 4x4 rotation matrices
    R_4x4 = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N_views, 1, 1)
    R_4x4[:, :, :3, :3] = rotation_matrices.unsqueeze(1).repeat(1, N_views, 1, 1)
    
    # Apply rotation: R * c2w (rotate the camera coordinate system)
    c2w_rotated = torch.matmul(R_4x4, c2w)
    
    return triangles_rotated, vn_rotated, c2w_rotated


class RenderFormerLoss(nn.Module):
    def __init__(self, lpips_weight=0.05, device='cuda'):
        """
        Combined loss function for RenderFormer training.
        
        Loss = L1(log(pred), log(gt)) + 0.05 * LPIPS(tone_map(pred), tone_map(gt))
        
        Args:
            lpips_weight: Weight for LPIPS loss (default: 0.05)
            device: Device for LPIPS network
        """
        super().__init__()
        self.lpips_weight = lpips_weight
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        self.device = device
        
        # Freeze LPIPS network
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
        
        # Initialize tone mapper
        self.use_tone_mapper = tone_mapper != 'none'
        if self.use_tone_mapper:
            if tone_mapper == 'pbr_neutral':
                tone_mapper = 'Khronos PBR Neutral'
            self.tone_mapper = ToneMapper(tone_mapper)
            print(f"Using {tone_mapper} tone mapper for LPIPS loss")
    
    def tone_map(self, hdr_img):
        """
        Tone mapping function using simple_ocio ToneMapper or simple clipping.
        
        Args:
            hdr_img: HDR image tensor [B, H, W, C]
        Returns:
            Tone-mapped image in [0, 1]
        """
        if self.use_tone_mapper:
            # Convert to numpy, apply tone mapping, convert back to torch
            # ToneMapper expects numpy arrays
            batch_size = hdr_img.shape[0]
            tone_mapped_imgs = []
            
            for i in range(batch_size):
                hdr_np = hdr_img[i].detach().cpu().numpy().astype(np.float32)
                ldr_np = self.tone_mapper.hdr_to_ldr(hdr_np)
                tone_mapped_imgs.append(torch.from_numpy(ldr_np))
            
            tone_mapped = torch.stack(tone_mapped_imgs).to(self.device)
        else:
            # Simple clipping fallback
            tone_mapped = torch.clamp(hdr_img, 0, 1)
        
        return tone_mapped
    
    def forward(self, pred, gt):
        """
        Compute combined loss.
        
        Args:
            pred: Predicted HDR images [B, N, H, W, 3]
            gt: Ground truth HDR images [B, N, H, W, 3]
        Returns:
            Combined loss value
        """
        # Reshape to combine batch and view dimensions
        B, N, H, W, C = pred.shape
        pred_flat = pred.reshape(B * N, H, W, C)
        gt_flat = gt.reshape(B * N, H, W, C)
        
        # L1 loss on log-transformed images
        log_pred = torch.log10(pred_flat + 1.0)
        log_gt = torch.log10(gt_flat + 1.0)
        l1_loss = nn.functional.l1_loss(log_pred, log_gt)
        
        # LPIPS loss on tone-mapped images
        tone_pred = self.tone_map(pred_flat)
        tone_gt = self.tone_map(gt_flat)
        
        # LPIPS expects [B, C, H, W] format
        tone_pred = tone_pred.permute(0, 3, 1, 2)
        tone_gt = tone_gt.permute(0, 3, 1, 2)
        
        lpips_loss = self.lpips_fn(tone_pred, tone_gt).mean()
        
        # Combined loss
        total_loss = l1_loss + self.lpips_weight * lpips_loss
        
        return total_loss, l1_loss, lpips_loss


def train_epoch(pipeline, dataloader, optimizer, scheduler, criterion, device, resolution, precision, use_rotation_aug=True):
    pipeline.model.train()
    total_loss = 0
    total_l1 = 0
    total_lpips = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        triangles = batch['triangles'].to(device)
        texture = batch['texture'].to(device)
        mask = batch['mask'].to(device)
        vn = batch['vn'].to(device)
        c2w = batch['c2w'].to(device)
        fov = batch['fov'].unsqueeze(-1).to(device)
        gt_images = batch['gt_images'].to(device)
        
        # Apply random rotation augmentation
        if use_rotation_aug:
            triangles, vn, c2w = random_rotation_augmentation(triangles, vn, c2w)
        
        optimizer.zero_grad()
        
        pred_images = pipeline(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=torch.float16 if precision == 'fp16' else torch.bfloat16 if precision == 'bf16' else torch.float32,
        )
        
        loss, l1_loss, lpips_loss = criterion(pred_images, gt_images)
        
        loss.backward()
        optimizer.step()
        
        # Step scheduler per batch (not per epoch!)
        scheduler.step()
        
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_lpips += lpips_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'lpips': f'{lpips_loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_l1 = total_l1 / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    
    return avg_loss, avg_l1, avg_lpips

@torch.no_grad()
def validate(pipeline, dataloader, criterion, device, resolution, precision):
    pipeline.model.eval()
    total_loss = 0
    total_l1 = 0
    total_lpips = 0
    
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        triangles = batch['triangles'].to(device)
        texture = batch['texture'].to(device)
        mask = batch['mask'].to(device)
        vn = batch['vn'].to(device)
        c2w = batch['c2w'].to(device)
        fov = batch['fov'].unsqueeze(-1).to(device)
        gt_images = batch['gt_images'].to(device)
        
        pred_images = pipeline(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=torch.float16 if precision == 'fp16' else torch.bfloat16 if precision == 'bf16' else torch.float32,
        )
        
        loss, l1_loss, lpips_loss = criterion(pred_images, gt_images)
        
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_lpips += lpips_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'lpips': f'{lpips_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_l1 = total_l1 / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    
    return avg_loss, avg_l1, avg_lpips

def main():
    parser = argparse.ArgumentParser(description="Train RenderFormer model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model_id", type=str, default=None, help="Pretrained model ID (optional)")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16', help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--padding_length", type=int, default=None, help="Padding length for triangles (optional)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=8000, help="Number of warmup steps")
    parser.add_argument("--lpips_weight", type=float, default=0.05, help="Weight for LPIPS loss")
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='agx', help="Tone mapper for LPIPS loss")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--no_rotation_aug", action='store_true', help="Disable rotation augmentation")
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    if args.model_id:
        print(f"Loading pretrained model from {args.model_id}")
        pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    else:
        print("Initializing model from scratch")
        pipeline = RenderFormerRenderingPipeline(RenderFormer(RenderFormerConfig()))
    
    # Apply optimizations for CUDA
    if device.type == 'cuda' and os.name == 'posix':
        from renderformer_liger_kernel import apply_kernels
        apply_kernels(pipeline.model)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled TF32 for matmul and cuDNN")
    elif device == torch.device('mps'):
        args.precision = 'fp32'
        print("bf16 and fp16 will cause too large error in MPS, force using fp32 instead.")
    
    pipeline.to(device)
    
    # Create datasets
    train_dataset = RenderFormerTrainingDataset(
        os.path.join(args.data_dir, 'train'), 
        padding_length=args.padding_length
    )
    val_dataset = RenderFormerTrainingDataset(
        os.path.join(args.data_dir, 'val'), 
        padding_length=args.padding_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Loss function
    criterion = RenderFormerLoss(
        lpips_weight=args.lpips_weight,
        tone_mapper=args.tone_mapper,
        device=device
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        pipeline.model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler: Linear warmup + Cosine decay
    # Using PyTorch's built-in schedulers
    total_steps = args.epochs * len(train_loader)
    
    # Warmup scheduler: linear from 0 to peak lr
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,  # Start from very small LR
        end_factor=1e-4,      # End at peak LR (args.lr)
        total_iters=args.warmup_steps
    )
    
    # Cosine annealing scheduler: decay from peak lr to 0
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=0.0
    )
    
    # Sequential scheduler: warmup then cosine
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )
    
    # Print training configuration
    use_rotation_aug = not args.no_rotation_aug
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Peak learning rate: {args.lr}")
    print(f"  Rotation augmentation: {'enabled' if use_rotation_aug else 'disabled'}")
    print(f"  Precision: {args.precision}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_l1, train_lpips = train_epoch(
            pipeline, train_loader, optimizer, scheduler, criterion, device, 
            args.resolution, args.precision, use_rotation_aug
        )
        
        # Validate
        val_loss, val_l1, val_lpips = validate(
            pipeline, val_loader, criterion, device, args.resolution, args.precision
        )
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} (L1: {train_l1:.4f}, LPIPS: {train_lpips:.4f})")
        print(f"Val Loss: {val_loss:.4f} (L1: {val_l1:.4f}, LPIPS: {val_lpips:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': pipeline.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': pipeline.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()