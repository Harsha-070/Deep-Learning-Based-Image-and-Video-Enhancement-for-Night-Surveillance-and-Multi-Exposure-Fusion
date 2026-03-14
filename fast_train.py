"""Fast resume training — picks up from saved best_model.pth with smaller crops."""
import os
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Unbuffered output so progress prints immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import config  # noqa: E402 — must come after stdout redirect
from models import EnhanceNet
from utils.losses import CombinedLoss
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim

torch.set_num_threads(os.cpu_count() or 4)

# ── Fast settings: small crops = ~4x faster per epoch ──────────────────────
EPOCHS = 35
BATCH_SIZE = 4
IMAGE_SIZE = 64
LR = 5e-5

config.BATCH_SIZE = BATCH_SIZE
config.IMAGE_SIZE = IMAGE_SIZE
config.W_PERCEPTUAL = 0.0
config.W_RECONSTRUCTION = 8.0

device = config.DEVICE
print(f"=== FAST RESUME TRAINING: {EPOCHS} epochs, {IMAGE_SIZE}px, BS={BATCH_SIZE} ===")

# ── Load checkpoint ─────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(config.PRETRAINED_DIR, 'best_model.pth')
if not os.path.exists(CKPT_PATH):
    print(f"\n  ERROR: No checkpoint found at '{CKPT_PATH}'.")
    print("  Run 'python run_train.py' first to train from scratch.")
    sys.exit(1)

model = EnhanceNet(in_channels=3, hidden_channels=32, num_curves=6).to(device)
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
prev_psnr = checkpoint.get('psnr', 0.0)
prev_epoch = checkpoint.get('epoch', 0)
print(f"Resumed from epoch {prev_epoch}, PSNR={prev_psnr:.2f} dB")

# ── Optimizer / scheduler / data ────────────────────────────────────────────
criterion = CombinedLoss(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
train_loader, test_loader = get_dataloaders(config)

best_psnr = prev_psnr
os.makedirs(config.PRETRAINED_DIR, exist_ok=True)
start = time.time()

# ── Training loop ────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for low, high, _ in train_loader:
        low, high = low.to(device), high.to(device)
        enhanced, curve_params, _ = model(low)
        loss, _ = criterion(enhanced, low, curve_params, target=high)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / max(batch_count, 1)

    # Validate every 5 epochs and on first/last epoch
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for low, high, _ in test_loader:
                low, high = low.to(device), high.to(device)
                enhanced, _, _ = model(low)
                enh_np = enhanced.cpu().numpy().transpose(0, 2, 3, 1)
                tgt_np = high.cpu().numpy().transpose(0, 2, 3, 1)
                for enh, tgt in zip(enh_np, tgt_np):
                    psnr_list.append(calculate_psnr(enh, tgt))
                    ssim_list.append(calculate_ssim(enh, tgt))

        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)
        elapsed = time.time() - start
        print(
            f"  Ep {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | {elapsed:.0f}s"
        )
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    'epoch': prev_epoch + epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': best_psnr,
                    'ssim': val_ssim,
                },
                os.path.join(config.PRETRAINED_DIR, 'best_model.pth'),
            )
            print(f"  >>> NEW BEST! PSNR: {best_psnr:.2f} dB")
    else:
        elapsed = time.time() - start
        print(f"  Ep {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | {elapsed:.0f}s")

    scheduler.step()

# ── Save final checkpoint ────────────────────────────────────────────────────
torch.save(
    {
        'epoch': prev_epoch + EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr': best_psnr,
    },
    os.path.join(config.PRETRAINED_DIR, 'final_model.pth'),
)

total = time.time() - start
print(f"\n=== DONE! Best PSNR: {best_psnr:.2f} dB | Time: {total / 60:.1f} min ===")
