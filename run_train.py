"""
Lightweight training script optimized for CPU systems.
Trains Zero-DCE model on LOL dataset with reduced settings.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms.functional as TF

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print("=" * 60)
print("  Zero-DCE Training (CPU-Optimized)")
print("=" * 60)
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")

import config
from models import EnhanceNet
from utils.losses import CombinedLoss
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim

# CPU thread optimization
torch.set_num_threads(os.cpu_count() or 4)

# Settings - optimized for CPU
EPOCHS = 100
BATCH_SIZE = 2
IMAGE_SIZE = 128
LR = 1e-4

if torch.cuda.is_available():
    BATCH_SIZE = 8
    IMAGE_SIZE = 512

config.BATCH_SIZE = BATCH_SIZE
config.IMAGE_SIZE = IMAGE_SIZE
config.W_PERCEPTUAL = 0.0   # Disable VGG to save ~500MB RAM and ~40% compute
config.W_RECONSTRUCTION = 8.0  # Compensate for removed perceptual loss

device = config.DEVICE
print(f"  Device: {device}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Image Size: {IMAGE_SIZE}")
print("=" * 60)

# Model
print("\nCreating model...")
model = EnhanceNet(
    in_channels=config.INPUT_CHANNELS,
    hidden_channels=config.HIDDEN_CHANNELS,
    num_curves=config.NUM_CURVES,
).to(device)
print(f"  Parameters: {model.get_num_params():,}")

# Loss
print("Creating loss function...")
criterion = CombinedLoss(config).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Data
print("Loading dataset...")
if not os.path.exists(config.TRAIN_LOW_DIR) or not os.path.exists(config.TRAIN_HIGH_DIR):
    print(f"\n  ERROR: Training dataset not found at {config.DATASET_DIR}")
    print(f"  Expected directories:")
    print(f"    {config.TRAIN_LOW_DIR}")
    print(f"    {config.TRAIN_HIGH_DIR}")
    print(f"  Please download the LOL dataset first:")
    print(f"    python download_dataset.py")
    sys.exit(1)
train_loader, test_loader = get_dataloaders(config)

# Training
best_psnr = 0.0
os.makedirs(config.PRETRAINED_DIR, exist_ok=True)

print(f"\nStarting training...\n")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for low, high, _ in train_loader:
        low = low.to(device)
        high = high.to(device)

        enhanced, curve_params, _ = model(low)
        loss, loss_dict = criterion(enhanced, low, curve_params, target=high)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / max(batch_count, 1)

    # Validate every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for low, high, _ in test_loader:
                low = low.to(device)
                high = high.to(device)
                enhanced, _, _ = model(low)

                # Metrics
                enh_np = enhanced.cpu().numpy().transpose(0, 2, 3, 1)
                tgt_np = high.cpu().numpy().transpose(0, 2, 3, 1)

                for e, t in zip(enh_np, tgt_np):
                    val_psnr_list.append(calculate_psnr(e, t))
                    val_ssim_list.append(calculate_ssim(e, t))

        val_psnr = np.mean(val_psnr_list)
        val_ssim = np.mean(val_ssim_list)
        elapsed = time.time() - start_time

        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.0f}s")

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'ssim': val_ssim,
            }, os.path.join(config.PRETRAINED_DIR, 'best_model.pth'))
            print(f"  >>> New best model! PSNR: {best_psnr:.2f} dB")
    else:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

    scheduler.step()

# Save final model
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'psnr': best_psnr,
}, os.path.join(config.PRETRAINED_DIR, 'final_model.pth'))

total_time = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"  Training Complete!")
print(f"  Total time: {total_time / 60:.1f} minutes")
print(f"  Best PSNR: {best_psnr:.2f} dB")
print(f"  Models saved to: {config.PRETRAINED_DIR}")
print(f"{'=' * 60}")
