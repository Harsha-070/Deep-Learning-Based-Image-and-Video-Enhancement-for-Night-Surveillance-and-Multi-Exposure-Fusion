"""
Evaluation metrics for image enhancement quality assessment.

Metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity) - optional
    - MAE (Mean Absolute Error)
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(enhanced, target):
    """
    Calculate PSNR between enhanced and target images.

    Args:
        enhanced: Enhanced image as numpy array [H, W, C] in range [0, 255] or [0, 1].
        target: Target image as numpy array [H, W, C].

    Returns:
        psnr_value: PSNR in dB.
    """
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)
    if target.max() <= 1.0:
        target = (target * 255).astype(np.uint8)

    return peak_signal_noise_ratio(target, enhanced)


def calculate_ssim(enhanced, target):
    """
    Calculate SSIM between enhanced and target images.

    Args:
        enhanced: Enhanced image as numpy array [H, W, C] in range [0, 255] or [0, 1].
        target: Target image as numpy array [H, W, C].

    Returns:
        ssim_value: SSIM score in range [0, 1].
    """
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)
    if target.max() <= 1.0:
        target = (target * 255).astype(np.uint8)

    return structural_similarity(target, enhanced, channel_axis=2)


def calculate_mae(enhanced, target):
    """
    Calculate Mean Absolute Error between images.

    Args:
        enhanced: Enhanced image as numpy array.
        target: Target image as numpy array.

    Returns:
        mae_value: Mean absolute error.
    """
    if enhanced.max() > 1.0:
        enhanced = enhanced.astype(np.float32) / 255.0
    if target.max() > 1.0:
        target = target.astype(np.float32) / 255.0

    return np.mean(np.abs(enhanced.astype(np.float32) - target.astype(np.float32)))


def evaluate_batch(enhanced_batch, target_batch):
    """
    Evaluate a batch of enhanced images against ground truth.

    Args:
        enhanced_batch: Tensor [B, 3, H, W] or list of numpy arrays.
        target_batch: Tensor [B, 3, H, W] or list of numpy arrays.

    Returns:
        metrics: Dictionary with average PSNR and SSIM.
    """
    if isinstance(enhanced_batch, torch.Tensor):
        enhanced_batch = enhanced_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if isinstance(target_batch, torch.Tensor):
        target_batch = target_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)

    psnr_values = []
    ssim_values = []

    for enh, tgt in zip(enhanced_batch, target_batch):
        psnr_values.append(calculate_psnr(enh, tgt))
        ssim_values.append(calculate_ssim(enh, tgt))

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
    }
