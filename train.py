import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lpips

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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
    def __init__(self, lpips_weight=0.05, tone_mapper='agx', device='cuda'):
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


def train_epoch(pipeline, dataloader, optimizer, scheduler, criterion, device, resolution, precision, use_rotation_aug=True, epoch=0):
    pipeline.model.train()
    total_loss = 0
    total_l1 = 0
    total_lpips = 0
    batch_count = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        triangles = batch['triangles'].to(device)
        texture = batch['texture'].to(device)
        mask = batch['mask'].to(device)
        vn = batch['vn'].to(device)
        c2w = batch['c2w'].to(device)
        fov = batch['fov'].unsqueeze(-1).to(device)
        gt_images = batch['gt_images'].to(device)
        scene_name = batch['scene_name']

        # first_param = next(p for p in pipeline.model.parameters() if p.requires_grad)
        # print("Initial param norm:", first_param.norm().item())
        # print("Triangles shape:", triangles.shape)
        # print("Texture shape:", texture.shape)
        # print("Mask shape:", mask.shape)
        # print("Vertex normals shape:", vn.shape)
        # print("Camera-to-world shape:", c2w.shape)
        # print("FOV shape:", fov.shape)
        # print("GT images shape:", gt_images.shape)

        # Visualize some ground truth images for debugging
        for i in range(gt_images.shape[0]):
            img = gt_images[i, 0].cpu().numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.savefig(f"debug/gt_epoch_{epoch}_batch_{batch_count}_sample_{i}.png")
            plt.close()

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
            enable_grad=True
        )

        # print("Pred images shape:", pred_images.shape)

        # visualize all predicted images in the first batch for debugging
        for i in range(pred_images.shape[0]):
            img = pred_images[i, 0].detach().cpu().float().numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.savefig(f"debug/pred_epoch_{epoch}_batch_{batch_count}_sample_{i}.png")
            plt.close()

        # pred_images: [B, N_views, H, W, 3], gt_images: [B, N_views, H, W, 3]
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
        # print("Final param norm:", first_param.norm().item())

        batch_count += 1
    
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
        scene_name = batch['scene_name']
        
        pred_images = pipeline(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            torch_dtype=torch.float16 if precision == 'fp16' else torch.bfloat16 if precision == 'bf16' else torch.float32,
            enable_grad=False
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

def print_debug_info(pipeline, device, args):
    """Print detailed information about the RenderFormer pipeline"""
    print("="*80)
    print("PIPELINE INFORMATION")
    print("="*80)
    
    # Basic info
    print(f"\nPipeline type: {type(pipeline).__name__}")
    print(f"Device: {device}")
    
    # Model info
    model = pipeline.model
    print(f"\nModel type: {type(model).__name__}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Model configuration
    if hasattr(model, 'config'):
        config = model.config
        print(f"\nModel Configuration:")
        for key, value in vars(config).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (fp32)")
    
    # Memory usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    # Layer breakdown
    print(f"\nModel Architecture:")
    total_layers = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            total_layers += 1
    print(f"  Total layers: {total_layers}")
    
    # Print first few layers
    print(f"\nFirst 10 layers:")
    for i, (name, module) in enumerate(model.named_modules()):
        if len(list(module.children())) == 0 and i < 10:
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {type(module).__name__} ({params:,} params)")
    
    # Module types summary
    print(f"\nModule Type Summary:")
    module_types = {}
    for module in model.modules():
        module_type = type(module).__name__
        if module_type not in module_types:
            module_types[module_type] = 0
        module_types[module_type] += 1
    
    for module_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {module_type}: {count}")
    
    print("="*80)

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Peak learning rate: {args.peak_lr}")
    print(f"  Rotation augmentation: {'enabled' if args.use_rotation_aug else 'disabled'}")
    print(f"  Precision: {args.precision}")

def main():
    # avaialable pretrained model IDs:
    # microsoft/renderformer-v1.1-swin-large
    # microsoft/renderformer-v1-base
    parser = argparse.ArgumentParser(description="Train RenderFormer model")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to Metadata JSON file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model_id", type=str, default=None, help="Pretrained model ID (optional)")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='fp16', help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--padding_length", type=int, default=4096, help="Padding length for triangles (optional)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--peak_lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=8000, help="Number of warmup steps")
    parser.add_argument("--lpips_weight", type=float, default=0.05, help="Weight for LPIPS loss")
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='agx', help="Tone mapper for LPIPS loss")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--use_rotation_aug", action='store_true', default=True, help="Enable rotation augmentation")
    parser.add_argument('-v', "--verbose", action='store_true', default=False, help="Print detailed model information")
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preparing for summary logging
    log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)

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

    if args.verbose:
        print_debug_info(pipeline, device, args)

    # Create datasets
    train_dataset = RenderFormerTrainingDataset(
        metadata_path=args.metadata_path,
        split='train',
        train_ratio=0.85,
        padding_length=args.padding_length
    )

    val_dataset = RenderFormerTrainingDataset(
        metadata_path=args.metadata_path,
        split='val',
        train_ratio=0.85,
        padding_length=args.padding_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
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
        lr=args.peak_lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler: Linear warmup + Cosine decay
    # total steps = epochs * num_batches
    total_steps = args.epochs * len(train_loader)
    
    # Warmup scheduler: linear from 0 to peak lr
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1,
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

    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_l1, train_lpips = train_epoch(
            pipeline, train_loader, optimizer, scheduler, criterion, device, 
            args.resolution, args.precision, args.use_rotation_aug, epoch=epoch,
        )
        
        # Validate
        val_loss, val_l1, val_lpips = validate(
            pipeline, val_loader, criterion, device, args.resolution, args.precision
        )
        
        lr = scheduler.get_last_lr()[0]

        # --- TensorBoard logging (per epoch) ---
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_L1", train_l1, epoch)
        writer.add_scalar("Loss/train_LPIPS", train_lpips, epoch)

        writer.add_scalar("Loss/val_total", val_loss, epoch)
        writer.add_scalar("Loss/val_L1", val_l1, epoch)
        writer.add_scalar("Loss/val_LPIPS", val_lpips, epoch)

        writer.add_scalar("LR/learning_rate", lr, epoch)
        # --------------------------------------

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