"""
Training script for the Zero-DCE Night Surveillance Enhancement model.

Supports:
    - Training from scratch
    - Fine-tuning with paired LOL dataset supervision
    - Automatic model checkpointing
    - Training progress visualization

Usage:
    python train.py                          # Train with default settings
    python train.py --epochs 100 --lr 1e-4   # Custom training
    python train.py --resume pretrained/best_model.pth  # Resume training
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from models import EnhanceNet
from utils.losses import CombinedLoss
from utils.dataloader import get_dataloaders
from utils.metrics import evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser(description='Train Zero-DCE Enhancement Model')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=config.PRETRAINED_DIR)
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    loss_components = {}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    for batch_idx, (low, high, _) in enumerate(pbar):
        low = low.to(device)
        high = high.to(device)

        # Forward pass
        enhanced, curve_params, curve_maps = model(low)

        # Compute loss (with paired supervision)
        loss, loss_dict = criterion(enhanced, low, curve_params, target=high)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        running_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rec': f'{loss_dict.get("reconstruction", 0):.4f}',
        })

    n = len(train_loader)
    avg_losses = {k: v / n for k, v in loss_components.items()}
    return running_loss / n, avg_losses


@torch.no_grad()
def validate(model, test_loader, criterion, device):
    """Validate on test set."""
    model.eval()
    total_loss = 0.0
    all_psnr = []
    all_ssim = []

    for low, high, _ in test_loader:
        low = low.to(device)
        high = high.to(device)

        enhanced, curve_params, _ = model(low)
        loss, _ = criterion(enhanced, low, curve_params, target=high)
        total_loss += loss.item()

        # Compute metrics
        metrics = evaluate_batch(enhanced, high)
        all_psnr.append(metrics['psnr'])
        all_ssim.append(metrics['ssim'])

    n = len(test_loader)
    return {
        'loss': total_loss / n,
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
    }


def save_training_curves(history, save_dir):
    """Save training loss and metric curves as plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # PSNR curve
    axes[1].plot(history['psnr'], color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Validation PSNR')
    axes[1].grid(True)

    # SSIM curve
    axes[2].plot(history['ssim'], color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM')
    axes[2].set_title('Validation SSIM')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"  Training curves saved to {save_dir}/training_curves.png")


def main():
    args = parse_args()

    # Setup
    device = config.DEVICE
    os.makedirs(args.save_dir, exist_ok=True)
    print("=" * 60)
    print("  Zero-DCE Night Surveillance Enhancement - Training")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Image Size  : {args.image_size}")
    print("=" * 60)

    # Update config with CLI args
    config.BATCH_SIZE = args.batch_size
    config.IMAGE_SIZE = args.image_size

    # Model
    model = EnhanceNet(
        in_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_curves=config.NUM_CURVES,
    ).to(device)
    print(f"\n  Model parameters: {model.get_num_params():,}")

    # Loss and optimizer
    criterion = CombinedLoss(config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_psnr = checkpoint.get('psnr', 0.0)   # preserve best so far
        print(f"  Resumed from {args.resume} (epoch {start_epoch}, best PSNR {best_psnr:.2f} dB)")

    if start_epoch >= args.epochs:
        print(f"  WARNING: start_epoch ({start_epoch}) >= epochs ({args.epochs}). "
              f"Increase --epochs to continue training.")
        return

    # Verify dataset exists
    if not os.path.exists(config.TRAIN_LOW_DIR) or not os.path.exists(config.TRAIN_HIGH_DIR):
        print(f"  ERROR: Training dataset not found at {config.DATASET_DIR}")
        print(f"  Please download the LOL dataset first: python download_dataset.py")
        return

    # Data loaders
    train_loader, test_loader = get_dataloaders(config)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'psnr': [], 'ssim': [],
    }

    print(f"\n  Starting training...\n")
    start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_loss, loss_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['psnr'].append(val_metrics['psnr'])
        history['ssim'].append(val_metrics['ssim'])

        # Print progress
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"PSNR: {val_metrics['psnr']:.2f} dB | "
              f"SSIM: {val_metrics['ssim']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.0f}s")

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': val_metrics['ssim'],
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  >>> New best model saved! PSNR: {best_psnr:.2f} dB")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim'],
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr': val_metrics['psnr'],
        'ssim': val_metrics['ssim'],
    }, os.path.join(args.save_dir, 'final_model.pth'))

    # Save training curves
    save_training_curves(history, args.save_dir)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Best PSNR : {best_psnr:.2f} dB")
    print(f"  Models saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
