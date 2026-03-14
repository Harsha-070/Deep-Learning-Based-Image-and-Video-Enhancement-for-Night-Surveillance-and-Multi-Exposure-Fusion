"""
Multi-Exposure Fusion (MEF) module for handling images with
mixed exposure regions (both dark and bright areas).

Approach:
    1. Generate multiple synthetic exposures from the input image
       using gamma correction and the Zero-DCE model
    2. Compute quality weight maps based on:
       - Contrast (Laplacian filter response)
       - Saturation (standard deviation of RGB channels)
       - Well-Exposedness (closeness to mid-range intensity)
    3. Fuse using Laplacian pyramid blending (Mertens' algorithm)

This handles real-world images with both underexposed and
overexposed regions in a single frame.

Usage:
    python multi_exposure_fusion.py --input path/to/image.jpg
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import config
from models import EnhanceNet


class MultiExposureFusion:
    """
    Multi-Exposure Fusion engine that combines multiple exposure
    levels to produce a well-balanced output image.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Path to trained Zero-DCE model (for generating exposures).
            device: torch device.
        """
        self.device = device or config.DEVICE

        # Load Zero-DCE model for intelligent exposure generation
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = EnhanceNet(
                in_channels=config.INPUT_CHANNELS,
                hidden_channels=config.HIDDEN_CHANNELS,
                num_curves=config.NUM_CURVES,
            ).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"  Zero-DCE model loaded for exposure generation.")

    def generate_exposures(self, image, gamma_values=None):
        """
        Generate multiple exposure versions of an image.

        Uses both gamma correction and Zero-DCE enhancement
        for intelligent exposure generation.

        Args:
            image: Input image as numpy array [H, W, 3] BGR uint8.
            gamma_values: List of gamma values for exposure simulation.

        Returns:
            exposures: List of exposure images [H, W, 3] BGR uint8.
        """
        if gamma_values is None:
            gamma_values = config.MEF_EXPOSURES

        exposures = []
        img_float = image.astype(np.float32) / 255.0

        for gamma in gamma_values:
            # Gamma correction for exposure simulation
            exposed = np.power(img_float, 1.0 / gamma)
            exposed = np.clip(exposed * 255, 0, 255).astype(np.uint8)
            exposures.append(exposed)

        # Add Zero-DCE enhanced version
        if self.model is not None:
            enhanced = self._model_enhance(image)
            exposures.append(enhanced)

        return exposures

    @torch.no_grad()
    def _model_enhance(self, image):
        """Apply Zero-DCE enhancement to generate an additional exposure."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Pad to multiple of 4
        pad_w = (4 - w % 4) % 4
        pad_h = (4 - h % 4) % 4
        if pad_w or pad_h:
            image_rgb = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        input_tensor = TF.to_tensor(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        enhanced_tensor, _, _ = self.model(input_tensor)

        enhanced = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced = enhanced[:h, :w, :]

        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    def compute_weight_maps(self, exposures):
        """
        Compute quality-based weight maps for each exposure.

        Weights are based on:
            1. Contrast: Response to Laplacian filter (edge detection)
            2. Saturation: Standard deviation of RGB channels
            3. Well-Exposedness: Gaussian around mid-intensity (0.5)

        Args:
            exposures: List of exposure images [H, W, 3] BGR uint8.

        Returns:
            weights: List of weight maps [H, W] float32, normalized.
        """
        weights = []

        for img in exposures:
            img_float = img.astype(np.float32) / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # Contrast weight (Laplacian filter response)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            contrast = np.abs(laplacian)

            # Saturation weight (std of RGB channels)
            saturation = np.std(img_float, axis=2)

            # Well-exposedness weight (Gaussian around 0.5)
            sigma = 0.2
            well_exposed = np.exp(-0.5 * ((img_float - 0.5) / sigma) ** 2)
            well_exposed = np.prod(well_exposed, axis=2)

            # Combined weight
            weight = (contrast + 1e-8) * (saturation + 1e-8) * (well_exposed + 1e-8)
            weights.append(weight)

        # Normalize weights
        weight_sum = sum(weights) + 1e-8
        weights = [w / weight_sum for w in weights]

        return weights

    def pyramid_blend(self, exposures, weights, levels=5):
        """
        Blend multiple exposures using Laplacian pyramid fusion.
        This is based on Mertens et al. (2007) exposure fusion.

        Args:
            exposures: List of exposure images [H, W, 3] BGR uint8.
            weights: List of weight maps [H, W] float32.
            levels: Number of pyramid levels.

        Returns:
            fused: Blended output image [H, W, 3] BGR uint8.
        """
        # Build Laplacian pyramids for each exposure
        exposure_pyramids = []
        for img in exposures:
            img_float = img.astype(np.float32) / 255.0
            pyramid = self._build_laplacian_pyramid(img_float, levels)
            exposure_pyramids.append(pyramid)

        # Build Gaussian pyramids for weights
        weight_pyramids = []
        for w in weights:
            pyramid = self._build_gaussian_pyramid(w, levels)
            weight_pyramids.append(pyramid)

        # Blend each level
        fused_pyramid = []
        for level in range(levels):
            fused_level = np.zeros_like(exposure_pyramids[0][level])
            for i in range(len(exposures)):
                w = weight_pyramids[i][level]
                if len(fused_level.shape) == 3:
                    w = np.stack([w] * 3, axis=2)
                fused_level += w * exposure_pyramids[i][level]
            fused_pyramid.append(fused_level)

        # Reconstruct from pyramid
        fused = self._reconstruct_from_pyramid(fused_pyramid)
        fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

        return fused

    def _build_gaussian_pyramid(self, image, levels):
        """Build Gaussian pyramid."""
        pyramid = [image.copy()]
        current = image.copy()
        for _ in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _build_laplacian_pyramid(self, image, levels):
        """Build Laplacian pyramid."""
        gaussian = self._build_gaussian_pyramid(image, levels)
        laplacian = []
        for i in range(levels - 1):
            h, w = gaussian[i].shape[:2]
            upsampled = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
            laplacian.append(gaussian[i] - upsampled)
        laplacian.append(gaussian[-1])
        return laplacian

    def _reconstruct_from_pyramid(self, pyramid):
        """Reconstruct image from Laplacian pyramid."""
        image = pyramid[-1].copy()
        for i in range(len(pyramid) - 2, -1, -1):
            h, w = pyramid[i].shape[:2]
            image = cv2.pyrUp(image, dstsize=(w, h))
            image = image + pyramid[i]
        return image

    def fuse(self, image, gamma_values=None):
        """
        Complete multi-exposure fusion pipeline.

        Args:
            image: Input image [H, W, 3] BGR uint8.
            gamma_values: Optional gamma values for exposure generation.

        Returns:
            fused: Fused output image [H, W, 3] BGR uint8.
        """
        # Generate multiple exposures
        exposures = self.generate_exposures(image, gamma_values)

        # Compute weight maps
        weights = self.compute_weight_maps(exposures)

        # Pyramid blending
        fused = self.pyramid_blend(exposures, weights)

        # Post-processing: bilateral denoising to remove noise amplified by fusion
        fused = cv2.bilateralFilter(fused, d=9, sigmaColor=75, sigmaSpace=75)

        return fused

    def fuse_opencv_mertens(self, image, gamma_values=None):
        """
        Alternative fusion using OpenCV's Mertens algorithm.

        Args:
            image: Input image [H, W, 3] BGR uint8.
            gamma_values: Optional gamma values.

        Returns:
            fused: Fused output image [H, W, 3] BGR uint8.
        """
        exposures = self.generate_exposures(image, gamma_values)

        # OpenCV Mertens fusion
        merge_mertens = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0,
        )
        fused = merge_mertens.process(exposures)
        fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

        return fused

    def create_exposure_strip(self, image, gamma_values=None):
        """
        Create a visual strip showing all generated exposures.

        Args:
            image: Input image [H, W, 3] BGR uint8.

        Returns:
            strip: Concatenated exposure images.
        """
        exposures = self.generate_exposures(image, gamma_values)

        # Resize all to same height
        target_h = 300
        resized = []
        for exp in exposures:
            scale = target_h / exp.shape[0]
            w = int(exp.shape[1] * scale)
            resized.append(cv2.resize(exp, (w, target_h)))

        separator = np.ones((target_h, 2, 3), dtype=np.uint8) * 255
        strips = []
        for i, img in enumerate(resized):
            if i > 0:
                strips.append(separator)
            strips.append(img)

        return np.hstack(strips)


def main():
    parser = argparse.ArgumentParser(description='Multi-Exposure Fusion')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='results/images', help='Output directory')
    parser.add_argument('--model', '-m', default='pretrained/best_model.pth', help='Model path')
    parser.add_argument('--method', choices=['pyramid', 'mertens'], default='pyramid',
                        help='Fusion method')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Multi-Exposure Fusion")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"  ERROR: Input image not found at '{args.input}'.")
        return

    fuser = MultiExposureFusion(model_path=args.model)

    image = cv2.imread(args.input)
    if image is None:
        print(f"  ERROR: Could not read {args.input}")
        return

    # Fuse
    if args.method == 'pyramid':
        fused = fuser.fuse(image)
    else:
        fused = fuser.fuse_opencv_mertens(image)

    # Save results
    name = os.path.splitext(os.path.basename(args.input))[0]
    cv2.imwrite(os.path.join(args.output, f"{name}_fused.jpg"), fused)

    # Save exposure strip
    strip = fuser.create_exposure_strip(image)
    cv2.imwrite(os.path.join(args.output, f"{name}_exposures.jpg"), strip)

    # Save comparison
    h = max(image.shape[0], fused.shape[0])
    orig_resized = cv2.resize(image, (int(image.shape[1] * h / image.shape[0]), h))
    fused_resized = cv2.resize(fused, (int(fused.shape[1] * h / fused.shape[0]), h))
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 255
    comparison = np.hstack([orig_resized, separator, fused_resized])
    cv2.imwrite(os.path.join(args.output, f"{name}_mef_comparison.jpg"), comparison)

    print(f"\n  Results saved to {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
