"""
Evaluation script for the trained enhancement model.

Computes quantitative metrics on the LOL test set:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - MAE (Mean Absolute Error)

Generates visual comparison grids and per-image results.

Usage:
    python test.py
    python test.py --model pretrained/best_model.pth --save_visuals
"""

import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import config
from models import EnhanceNet
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mae


def evaluate_model(model, test_low_dir, test_high_dir, device, output_dir=None):
    """
    Evaluate model on paired test set.

    Args:
        model: Trained EnhanceNet model.
        test_low_dir: Directory with low-light test images.
        test_high_dir: Directory with normal-light test images.
        device: torch device.
        output_dir: Directory to save visual results (optional).

    Returns:
        results: Dictionary with per-image and average metrics.
    """
    model.eval()

    low_files = sorted([f for f in os.listdir(test_low_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    high_files = sorted([f for f in os.listdir(test_high_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    assert len(low_files) == len(high_files), \
        f"Mismatch: {len(low_files)} low vs {len(high_files)} high"

    all_psnr, all_ssim, all_mae = [], [], []
    per_image_results = []

    print(f"\n  Evaluating {len(low_files)} test images...")
    print(f"  {'Image':30s} {'PSNR (dB)':>10s} {'SSIM':>8s} {'MAE':>8s}")
    print(f"  {'-' * 58}")

    for low_file, high_file in zip(low_files, high_files):
        # Load images
        low_path = os.path.join(test_low_dir, low_file)
        high_path = os.path.join(test_high_dir, high_file)

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # Ensure same size
        w = min(low_img.size[0], high_img.size[0])
        h = min(low_img.size[1], high_img.size[1])
        low_img = low_img.resize((w, h), Image.LANCZOS)
        high_img = high_img.resize((w, h), Image.LANCZOS)

        # Pad for model
        pad_w = (4 - w % 4) % 4
        pad_h = (4 - h % 4) % 4
        low_np = np.array(low_img)
        if pad_w or pad_h:
            low_np = np.pad(low_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Enhance
        with torch.no_grad():
            input_tensor = TF.to_tensor(Image.fromarray(low_np)).unsqueeze(0).to(device)
            enhanced_tensor, _, _ = model(input_tensor)

        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced_np = enhanced_np[:h, :w, :]  # Remove padding
        enhanced_np = np.clip(enhanced_np, 0, 1)

        high_np = np.array(high_img).astype(np.float32) / 255.0

        # Compute metrics
        psnr = calculate_psnr(enhanced_np, high_np)
        ssim = calculate_ssim(enhanced_np, high_np)
        mae = calculate_mae(enhanced_np, high_np)

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_mae.append(mae)

        per_image_results.append({
            'filename': low_file,
            'psnr': psnr,
            'ssim': ssim,
            'mae': mae,
        })

        print(f"  {low_file:30s} {psnr:10.2f} {ssim:8.4f} {mae:8.4f}")

        # Save visual comparison
        if output_dir:
            save_comparison(
                np.array(low_img), enhanced_np, np.array(high_img),
                os.path.join(output_dir, f"eval_{os.path.splitext(low_file)[0]}.png")
            )

    # Average metrics
    avg_metrics = {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'mae': np.mean(all_mae),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim),
    }

    print(f"  {'-' * 58}")
    print(f"  {'AVERAGE':30s} {avg_metrics['psnr']:10.2f} {avg_metrics['ssim']:8.4f} {avg_metrics['mae']:8.4f}")
    print(f"  {'STD DEV':30s} {avg_metrics['psnr_std']:10.2f} {avg_metrics['ssim_std']:8.4f}")

    return {
        'per_image': per_image_results,
        'average': avg_metrics,
    }


def save_comparison(low, enhanced, high, save_path):
    """Save a 3-panel comparison: Input | Enhanced | Ground Truth."""
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)
    if high.max() <= 1.0:
        high = (high * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(low)
    axes[0].set_title('Input (Low-Light)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(enhanced)
    axes[1].set_title('Enhanced (Ours)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(high)
    axes[2].set_title('Ground Truth', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_results_grid(test_low_dir, test_high_dir, model, device, save_path, num_images=5):
    """Create a grid of results for the paper/report."""
    model.eval()

    low_files = sorted([f for f in os.listdir(test_low_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])[:num_images]

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, low_file in enumerate(low_files):
        # Load
        low_img = Image.open(os.path.join(test_low_dir, low_file)).convert('RGB')
        high_files_list = sorted([f for f in os.listdir(test_high_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        high_img = Image.open(os.path.join(test_high_dir, high_files_list[idx])).convert('RGB')

        w = min(low_img.size[0], high_img.size[0])
        h = min(low_img.size[1], high_img.size[1])
        low_img = low_img.resize((w, h), Image.LANCZOS)
        high_img = high_img.resize((w, h), Image.LANCZOS)

        # Enhance
        low_np = np.array(low_img)
        pad_w = (4 - w % 4) % 4
        pad_h = (4 - h % 4) % 4
        if pad_w or pad_h:
            low_padded = np.pad(low_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            low_padded = low_np

        with torch.no_grad():
            input_tensor = TF.to_tensor(Image.fromarray(low_padded)).unsqueeze(0).to(device)
            enhanced_tensor, _, _ = model(input_tensor)

        enhanced = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = enhanced[:h, :w, :]
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Plot
        axes[idx, 0].imshow(low_np)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title('Input (Low-Light)', fontsize=14, fontweight='bold')

        axes[idx, 1].imshow(enhanced)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title('Enhanced (Ours)', fontsize=14, fontweight='bold')

        axes[idx, 2].imshow(np.array(high_img))
        axes[idx, 2].axis('off')
        if idx == 0:
            axes[idx, 2].set_title('Ground Truth', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Results grid saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate enhancement model')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth', help='Model path')
    parser.add_argument('--save_visuals', action='store_true', help='Save comparison images')
    parser.add_argument('--output', '-o', default='results/evaluation', help='Output directory')
    args = parser.parse_args()

    device = config.DEVICE
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Model  : {args.model}")

    # Load model
    model = EnhanceNet(
        in_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_curves=config.NUM_CURVES,
    ).to(device)

    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model loaded successfully.")
        if 'psnr' in checkpoint:
            print(f"  Training best PSNR: {checkpoint['psnr']:.2f} dB")
    else:
        print(f"  ERROR: Model not found at '{args.model}'.")
        print(f"  Please train the model first: python run_train.py")
        return

    # Verify test dataset exists
    if not os.path.exists(config.TEST_LOW_DIR) or not os.path.exists(config.TEST_HIGH_DIR):
        print(f"  ERROR: Test dataset not found.")
        print(f"  Please download the LOL dataset first: python download_dataset.py")
        return

    # Evaluate
    output_dir = args.output if args.save_visuals else None
    results = evaluate_model(
        model, config.TEST_LOW_DIR, config.TEST_HIGH_DIR,
        device, output_dir
    )

    # Create results grid
    create_results_grid(
        config.TEST_LOW_DIR, config.TEST_HIGH_DIR,
        model, device,
        os.path.join(args.output, 'results_grid.png'),
        num_images=min(5, len(os.listdir(config.TEST_LOW_DIR)))
    )

    # Save metrics to file
    with open(os.path.join(args.output, 'metrics.txt'), 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-Image Results:\n")
        f.write(f"{'Image':30s} {'PSNR':>8s} {'SSIM':>8s} {'MAE':>8s}\n")
        for r in results['per_image']:
            f.write(f"{r['filename']:30s} {r['psnr']:8.2f} {r['ssim']:8.4f} {r['mae']:8.4f}\n")
        f.write(f"\nAverage Metrics:\n")
        f.write(f"  PSNR: {results['average']['psnr']:.2f} +/- {results['average']['psnr_std']:.2f} dB\n")
        f.write(f"  SSIM: {results['average']['ssim']:.4f} +/- {results['average']['ssim_std']:.4f}\n")
        f.write(f"  MAE:  {results['average']['mae']:.4f}\n")

    print(f"\n  Metrics saved to {args.output}/metrics.txt")
    print("=" * 60)


if __name__ == '__main__':
    main()
