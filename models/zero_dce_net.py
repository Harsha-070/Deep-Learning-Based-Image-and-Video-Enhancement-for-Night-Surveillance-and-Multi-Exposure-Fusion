"""
Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.

Based on: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
by Chunle Guo et al. (CVPR 2020).

The model learns a set of best-fitting Light Enhancement curves (LE-curves)
to adjust the dynamic range of a given image. The curve estimation is done
pixel-wise and is guided by a set of non-reference loss functions.

Enhancement formula (applied iteratively):
    LE(x) = x + alpha * x * (1 - x)

where alpha is the learned curve parameter and x is the pixel intensity.
"""

import torch
import torch.nn as nn


class ZeroDCENet(nn.Module):
    """
    Zero-DCE network for estimating pixel-wise curve parameters.

    Architecture:
        - 7 convolutional layers with ReLU activation
        - Symmetric skip connections between encoder and decoder layers
        - Output: 3 * n_iterations curve parameter maps

    The network is lightweight (~80K parameters) and efficient for
    real-time video processing.
    """

    def __init__(self, in_channels=3, hidden_channels=32, num_curves=6):
        """
        Args:
            in_channels: Number of input channels (3 for RGB).
            hidden_channels: Number of hidden feature channels.
            num_curves: Number of curve iterations (default: 8).
        """
        super(ZeroDCENet, self).__init__()
        self.num_curves = num_curves

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        # Decoder layers with skip connections
        self.conv5 = nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1)
        self.conv6 = nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1)

        # Output layer: produces 3 * num_curves channels (curve params per RGB per iteration)
        self.conv7 = nn.Conv2d(hidden_channels * 2, in_channels * num_curves, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass to estimate curve parameters.

        Args:
            x: Input image tensor [B, 3, H, W] in range [0, 1].

        Returns:
            curve_params: Curve parameter maps [B, 3*num_curves, H, W] in range [-1, 1].
        """
        # Encoder
        f1 = self.relu(self.conv1(x))     # [B, 32, H, W]
        f2 = self.relu(self.conv2(f1))    # [B, 32, H, W]
        f3 = self.relu(self.conv3(f2))    # [B, 32, H, W]
        f4 = self.relu(self.conv4(f3))    # [B, 32, H, W]

        # Decoder with symmetric skip connections
        f5 = self.relu(self.conv5(torch.cat([f4, f3], dim=1)))  # [B, 32, H, W]
        f6 = self.relu(self.conv6(torch.cat([f5, f2], dim=1)))  # [B, 32, H, W]

        # Output curve parameters
        curve_params = self.tanh(self.conv7(torch.cat([f6, f1], dim=1)))

        return curve_params


class EnhanceNet(nn.Module):
    """
    Complete enhancement network that combines Zero-DCE curve estimation
    with iterative curve application.

    This module wraps ZeroDCENet and applies the estimated curves to
    produce the enhanced image.
    """

    def __init__(self, in_channels=3, hidden_channels=32, num_curves=6):
        super(EnhanceNet, self).__init__()
        self.num_curves = num_curves
        self.curve_net = ZeroDCENet(in_channels, hidden_channels, num_curves)

    def apply_curves(self, image, curve_params):
        """
        Apply the Light Enhancement curves iteratively.

        Formula: LE(x) = x + alpha * x * (1 - x)
        Applied num_curves times with different alpha values.

        Args:
            image: Input image [B, 3, H, W] in range [0, 1].
            curve_params: Curve parameters [B, 3*num_curves, H, W].

        Returns:
            enhanced: Enhanced image [B, 3, H, W].
            curve_maps: List of alpha maps for each iteration.
        """
        enhanced = image
        curve_maps = []

        for i in range(self.num_curves):
            alpha = curve_params[:, i * 3:(i + 1) * 3, :, :]
            # LE curve: x + alpha * x * (1 - x)
            enhanced = enhanced + alpha * (enhanced - enhanced * enhanced)
            curve_maps.append(alpha)

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced, curve_maps

    def forward(self, x):
        """
        Full forward pass: estimate curves and apply them.

        Args:
            x: Input low-light image [B, 3, H, W] in range [0, 1].

        Returns:
            enhanced: Enhanced image [B, 3, H, W].
            curve_params: Raw curve parameters from the network.
            curve_maps: List of alpha maps per iteration.
        """
        curve_params = self.curve_net(x)
        enhanced, curve_maps = self.apply_curves(x, curve_params)
        return enhanced, curve_params, curve_maps

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
