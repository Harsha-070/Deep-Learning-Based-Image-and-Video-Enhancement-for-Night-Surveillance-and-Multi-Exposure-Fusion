"""
Video-Based Fine-Tuning for Night Surveillance Enhancement.

Extends the trained Zero-DCE model by fine-tuning on real surveillance
video frames using self-supervised losses — NO ground truth needed.

Why video training helps:
    - LOL dataset has still photos; surveillance videos have motion blur,
      compression artifacts, and different noise patterns
    - Fine-tuning on your actual surveillance footage makes the model
      specialised for your specific cameras and conditions
    - Zero-DCE's self-supervised losses work on ANY dark video

Setup:
    1. Place dark surveillance videos in:  datasets/videos/
    2. Run:  python train_video.py

Supported video formats: .mp4 .avi .mov .mkv .wmv .flv

The script:
    - Loads existing pretrained/best_model.pth
    - Extracts frames from all videos in datasets/videos/
    - Fine-tunes for 50 epochs using Zero-DCE self-supervised losses
    - Optionally mixes in LOL image pairs for supervised signal
    - Saves improved weights back to pretrained/best_model.pth
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import config
from models import EnhanceNet
from utils.losses import CombinedLoss
from utils.dataloader import get_video_dataloader, get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim

torch.set_num_threads(os.cpu_count() or 4)

# ── Settings ──────────────────────────────────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 4
IMAGE_SIZE     = 128
LR             = 2e-5      # Low LR — fine-tuning, not training from scratch
MIX_LOL        = True      # Also use LOL image pairs if available
LOL_MIX_RATIO  = 0.3       # Fraction of batches from LOL (rest from video)

config.BATCH_SIZE     = BATCH_SIZE
config.IMAGE_SIZE     = IMAGE_SIZE
config.W_PERCEPTUAL   = 0.0
config.W_RECONSTRUCTION = 6.0

device = config.DEVICE

print("=" * 60)
print("  Zero-DCE Video Fine-Tuning")
print("=" * 60)
print(f"  Device     : {device}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch Size : {BATCH_SIZE}")
print(f"  Image Size : {IMAGE_SIZE}px")
print(f"  Mix LOL    : {MIX_LOL}")
print(f"  LR         : {LR}")
print("=" * 60)

# ── Check videos directory ────────────────────────────────────────────────────
video_dir = config.VIDEO_DIR
os.makedirs(video_dir, exist_ok=True)

video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_exts)]

if not video_files:
    print(f"\n  ERROR: No videos found in '{video_dir}'")
    print(f"  Place your dark surveillance videos there and re-run.")
    print(f"\n  Supported formats: {', '.join(video_exts)}")
    sys.exit(1)

print(f"\n  Found {len(video_files)} video file(s):")
for vf in video_files:
    size_mb = os.path.getsize(os.path.join(video_dir, vf)) / 1e6
    print(f"    {vf}  ({size_mb:.1f} MB)")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_path = os.path.join(config.PRETRAINED_DIR, 'best_model.pth')
if not os.path.exists(ckpt_path):
    print(f"\n  ERROR: No checkpoint at '{ckpt_path}'.")
    print("  Run 'python run_train.py' first to train the base model.")
    sys.exit(1)

model = EnhanceNet(
    in_channels=config.INPUT_CHANNELS,
    hidden_channels=config.HIDDEN_CHANNELS,
    num_curves=config.NUM_CURVES,
).to(device)

checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
prev_psnr  = checkpoint.get('psnr', 0.0)
prev_epoch = checkpoint.get('epoch', 0)
print(f"\n  Loaded checkpoint: epoch {prev_epoch}, PSNR {prev_psnr:.2f} dB")

# ── Loss / Optimizer / Scheduler ─────────────────────────────────────────────
criterion = CombinedLoss(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# ── Data loaders ─────────────────────────────────────────────────────────────
print("\n  Building video frame dataset...")
video_loader = get_video_dataloader(video_dir, config)
print(f"  Video batches per epoch: {len(video_loader)}")

lol_loader = None
if MIX_LOL and os.path.exists(config.TRAIN_LOW_DIR):
    lol_loader, _ = get_dataloaders(config)
    print(f"  LOL batches per epoch:   {len(lol_loader)}")
else:
    print("  LOL dataset not mixed (not found or disabled).")

# ── Training loop ─────────────────────────────────────────────────────────────
best_psnr = prev_psnr
os.makedirs(config.PRETRAINED_DIR, exist_ok=True)
start_time = time.time()

# LOL iterator (cycles through if shorter than video loader)
lol_iter = iter(lol_loader) if lol_loader else None

print(f"\n  Starting fine-tuning...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss  = 0.0
    batch_count = 0

    for frames, _ in video_loader:
        frames = frames.to(device)

        # Self-supervised pass on video frames (no ground truth)
        enhanced, curve_params, _ = model(frames)
        loss, _ = criterion(enhanced, frames, curve_params, target=None)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        batch_count += 1

        # Occasionally mix in a supervised LOL batch
        if lol_iter is not None and batch_count % max(1, int(1 / LOL_MIX_RATIO)) == 0:
            try:
                low, high, _ = next(lol_iter)
            except StopIteration:
                lol_iter = iter(lol_loader)
                low, high, _ = next(lol_iter)

            low, high = low.to(device), high.to(device)
            enhanced_lol, cp_lol, _ = model(low)
            lol_loss, _ = criterion(enhanced_lol, low, cp_lol, target=high)

            optimizer.zero_grad()
            lol_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss  += lol_loss.item()
            batch_count += 1

    avg_loss = epoch_loss / max(batch_count, 1)
    elapsed  = time.time() - start_time

    # Validate every 5 epochs on LOL test set (if available)
    if (epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS) and \
       os.path.exists(config.TEST_LOW_DIR):

        model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            _, test_loader = get_dataloaders(config)
            for low, high, _ in test_loader:
                low, high = low.to(device), high.to(device)
                enhanced, _, _ = model(low)
                enh_np = enhanced.cpu().numpy().transpose(0, 2, 3, 1)
                tgt_np = high.cpu().numpy().transpose(0, 2, 3, 1)
                for e, t in zip(enh_np, tgt_np):
                    psnr_list.append(calculate_psnr(e, t))
                    ssim_list.append(calculate_ssim(e, t))

        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)

        print(f"  Ep {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | {elapsed:.0f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch':                prev_epoch + epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr':                 best_psnr,
                'ssim':                 val_ssim,
            }, ckpt_path)
            print(f"  >>> NEW BEST! PSNR: {best_psnr:.2f} dB — saved to {ckpt_path}")
    else:
        print(f"  Ep {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | {elapsed:.0f}s")

    scheduler.step()

# ── Save final model ──────────────────────────────────────────────────────────
final_path = os.path.join(config.PRETRAINED_DIR, 'final_model.pth')
torch.save({
    'epoch':            prev_epoch + EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'psnr':             best_psnr,
}, final_path)

total = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"  Video Fine-Tuning Complete!")
print(f"  Total time : {total / 60:.1f} minutes")
print(f"  Best PSNR  : {best_psnr:.2f} dB")
print(f"  Model saved: {ckpt_path}")
print(f"{'=' * 60}")
