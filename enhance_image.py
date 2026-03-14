"""
Image Enhancement Pipeline for low-light and night surveillance images.

Features:
    - Single image enhancement using trained Zero-DCE model
    - Batch processing of image directories
    - Optional post-processing (denoising, sharpening, contrast)
    - Side-by-side comparison output

Usage:
    python enhance_image.py --input path/to/image.jpg
    python enhance_image.py --input path/to/folder/ --output results/images/
"""

import os
import argparse
import time
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

import config
from models import EnhanceNet


class ImageEnhancer:
    """
    Image enhancement engine using the trained Zero-DCE model
    with optional post-processing pipeline.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Path to trained model weights.
            device: torch device (auto-detected if None).
        """
        self.device = device or config.DEVICE
        print(f"  Using device: {self.device}")

        # Load model
        self.model = EnhanceNet(
            in_channels=config.INPUT_CHANNELS,
            hidden_channels=config.HIDDEN_CHANNELS,
            num_curves=config.NUM_CURVES,
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Model loaded from {model_path}")
            if 'psnr' in checkpoint:
                print(f"  Model PSNR: {checkpoint['psnr']:.2f} dB")
        else:
            print("  WARNING: No model weights loaded. Using random initialization.")

        self.model.eval()

    @torch.no_grad()
    def enhance(self, image, post_process=True):
        """
        Enhance a single image.

        Args:
            image: Input image as numpy array [H, W, 3] BGR (OpenCV format)
                   or PIL Image.
            post_process: Whether to apply post-processing.

        Returns:
            enhanced: Enhanced image as numpy array [H, W, 3] BGR uint8.
        """
        # Convert input
        if isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Input must be numpy array or PIL Image")

        original_size = pil_image.size  # (W, H)

        # Pad to multiple of 4 for model compatibility
        w, h = pil_image.size
        pad_w = (4 - w % 4) % 4
        pad_h = (4 - h % 4) % 4
        if pad_w or pad_h:
            pil_image = Image.fromarray(
                np.pad(np.array(pil_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            )

        # To tensor
        input_tensor = TF.to_tensor(pil_image).unsqueeze(0).to(self.device)

        # Model inference
        enhanced_tensor, _, _ = self.model(input_tensor)

        # To numpy
        enhanced = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        # Remove padding
        if pad_w or pad_h:
            enhanced = enhanced[:original_size[1], :original_size[0], :]

        # Post-processing
        if post_process:
            enhanced = self._post_process(enhanced)

        # RGB to BGR for OpenCV compatibility
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        return enhanced_bgr

    def _post_process(self, image):
        """
        Post-processing pipeline:
            1. Bilateral denoising (edge-preserving noise reduction)
            2. CLAHE for local contrast on luminance only
            3. Mild sharpening
        """
        # Denoising: bilateral filter (edge-preserving, no API compatibility issues)
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert to LAB for CLAHE (only enhance luminance, not color noise)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to luminance channel with conservative clip limit
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Mild sharpening (low weight to avoid re-amplifying noise)
        gaussian = cv2.GaussianBlur(result, (0, 0), 2)
        result = cv2.addWeighted(result, 1.15, gaussian, -0.15, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    def create_comparison(self, original, enhanced, orientation='horizontal'):
        """
        Create side-by-side comparison image.

        Args:
            original: Original image (BGR).
            enhanced: Enhanced image (BGR).
            orientation: 'horizontal' or 'vertical'.
        """
        h1, w1 = original.shape[:2]
        h2, w2 = enhanced.shape[:2]

        # Resize to same height/width
        if orientation == 'horizontal':
            if h1 != h2:
                enhanced = cv2.resize(enhanced, (w2, h1))
            separator = np.ones((h1, 3, 3), dtype=np.uint8) * 255
            comparison = np.hstack([original, separator, enhanced])
        else:
            if w1 != w2:
                enhanced = cv2.resize(enhanced, (w1, h2))
            separator = np.ones((3, w1, 3), dtype=np.uint8) * 255
            comparison = np.vstack([original, separator, enhanced])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (0, 0, 255), 2)
        if orientation == 'horizontal':
            cv2.putText(comparison, 'Enhanced', (w1 + 13, 30), font, 1, (0, 255, 0), 2)
        else:
            cv2.putText(comparison, 'Enhanced', (10, h1 + 33), font, 1, (0, 255, 0), 2)

        return comparison


def process_single_image(enhancer, input_path, output_dir, save_comparison=True):
    """Process a single image file."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"  ERROR: Could not read {input_path}")
        return

    start = time.time()
    enhanced = enhancer.enhance(image)
    elapsed = time.time() - start

    # Save enhanced image
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
    cv2.imwrite(output_path, enhanced)

    # Save comparison
    if save_comparison:
        comparison = enhancer.create_comparison(image, enhanced)
        comp_path = os.path.join(output_dir, f"{name}_comparison{ext}")
        cv2.imwrite(comp_path, comparison)

    print(f"  {filename:30s} -> {elapsed:.3f}s | {image.shape[1]}x{image.shape[0]}")
    return enhanced


def main():
    parser = argparse.ArgumentParser(description='Enhance low-light images')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', default='results/images', help='Output directory')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth', help='Model path')
    parser.add_argument('--no_postprocess', action='store_true', help='Disable post-processing')
    parser.add_argument('--no_comparison', action='store_true', help='Skip comparison images')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Night Surveillance Image Enhancement")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"  ERROR: Model weights not found at '{args.model}'.")
        print(f"  Please train the model first: python run_train.py")
        print(f"  Or specify a valid model path: --model <path>")
        return

    enhancer = ImageEnhancer(model_path=args.model)

    if os.path.isfile(args.input):
        # Single image
        process_single_image(enhancer, args.input, args.output, not args.no_comparison)
    elif os.path.isdir(args.input):
        # Batch processing
        image_files = sorted([
            f for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"\n  Processing {len(image_files)} images...")

        for f in image_files:
            process_single_image(
                enhancer, os.path.join(args.input, f),
                args.output, not args.no_comparison
            )
    else:
        print(f"  ERROR: {args.input} does not exist")
        return

    print(f"\n  Results saved to {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
