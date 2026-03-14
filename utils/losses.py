"""
Loss functions for training the Zero-DCE enhancement network.

Includes both zero-reference losses (no ground truth needed) and
supervised losses (for fine-tuning with paired data like LOL dataset).

Zero-Reference Losses:
    1. Spatial Consistency Loss - Preserves spatial structure
    2. Exposure Control Loss   - Controls average intensity
    3. Color Constancy Loss    - Balances RGB channels
    4. Illumination Smoothness Loss - Ensures smooth enhancement

Supervised Losses (for fine-tuning):
    5. L1 Reconstruction Loss  - Pixel-wise similarity
    6. SSIM Loss              - Structural similarity
    7. Perceptual Loss        - VGG feature matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialConsistencyLoss(nn.Module):
    """
    Preserves the difference of neighboring regions between input and output.
    Ensures enhancement doesn't distort spatial relationships.
    """

    def __init__(self):
        super().__init__()
        # Kernels for computing differences with 4 neighbors
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('weight_left', kernel_left)
        self.register_buffer('weight_right', kernel_right)
        self.register_buffer('weight_up', kernel_up)
        self.register_buffer('weight_down', kernel_down)
        self.pool = nn.AvgPool2d(4)

    def forward(self, enhanced, original):
        """
        Args:
            enhanced: Enhanced image [B, 3, H, W].
            original: Original input image [B, 3, H, W].
        """
        org_mean = torch.mean(original, dim=1, keepdim=True)
        enh_mean = torch.mean(enhanced, dim=1, keepdim=True)

        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)

        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        d_enh_left = F.conv2d(enh_pool, self.weight_left, padding=1)
        d_enh_right = F.conv2d(enh_pool, self.weight_right, padding=1)
        d_enh_up = F.conv2d(enh_pool, self.weight_up, padding=1)
        d_enh_down = F.conv2d(enh_pool, self.weight_down, padding=1)

        d_left = (d_org_left - d_enh_left) ** 2
        d_right = (d_org_right - d_enh_right) ** 2
        d_up = (d_org_up - d_enh_up) ** 2
        d_down = (d_org_down - d_enh_down) ** 2

        return d_left.mean() + d_right.mean() + d_up.mean() + d_down.mean()


class ExposureControlLoss(nn.Module):
    """
    Controls the overall exposure level of the enhanced image.
    Measures deviation from a well-exposed target mean intensity.
    """

    def __init__(self, patch_size=16, target_mean=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_mean = target_mean

    def forward(self, enhanced):
        """
        Args:
            enhanced: Enhanced image [B, 3, H, W].
        """
        mean_intensity = torch.mean(enhanced, dim=1, keepdim=True)
        pooled = self.pool(mean_intensity)
        return torch.mean((pooled - self.target_mean) ** 2)


class ColorConstancyLoss(nn.Module):
    """
    Ensures color balance by penalizing deviation between
    channel-wise mean intensities. Based on the Gray-World hypothesis.
    """

    def forward(self, enhanced):
        """
        Args:
            enhanced: Enhanced image [B, 3, H, W].
        """
        mean_rgb = torch.mean(enhanced, dim=[2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = (mr - mg) ** 2
        d_rb = (mr - mb) ** 2
        d_gb = (mg - mb) ** 2
        return torch.mean(d_rg + d_rb + d_gb)


class IlluminationSmoothnessLoss(nn.Module):
    """
    Ensures spatial smoothness of the curve parameter maps.
    Penalizes large gradients in the alpha maps.
    """

    def forward(self, curve_params):
        """
        Args:
            curve_params: Curve parameter tensor [B, C, H, W].
        """
        B, C, H, W = curve_params.shape
        # TV loss in both directions
        tv_h = torch.mean((curve_params[:, :, 1:, :] - curve_params[:, :, :-1, :]) ** 2)
        tv_w = torch.mean((curve_params[:, :, :, 1:] - curve_params[:, :, :, :-1]) ** 2)
        count = B * C * (H - 1) * W + B * C * H * (W - 1)
        return (tv_h + tv_w) / max(count, 1) * H * W


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss for supervised fine-tuning.
    SSIM measures structural similarity between two images.
    Gaussian window is cached for performance.
    """

    def __init__(self, window_size=7, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Pre-compute and cache the Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('window', window)

    def forward(self, enhanced, target):
        """
        Args:
            enhanced: Enhanced image [B, 3, H, W].
            target: Ground truth image [B, 3, H, W].
        """
        pad = self.window_size // 2

        mu1 = F.conv2d(enhanced, self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(enhanced ** 2, self.window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(enhanced * target, self.window, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for supervised fine-tuning.
    Compares high-level features between enhanced and target images.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:16])).eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, enhanced, target):
        """
        Args:
            enhanced: Enhanced image [B, 3, H, W].
            target: Ground truth image [B, 3, H, W].
        """
        enh_norm = (enhanced - self.mean) / self.std
        tgt_norm = (target - self.mean) / self.std
        enh_features = self.features(enh_norm)
        tgt_features = self.features(tgt_norm)
        return F.l1_loss(enh_features, tgt_features)


class CombinedLoss(nn.Module):
    """
    Combined loss function for training Zero-DCE with both
    zero-reference and supervised components.

    The loss balances self-supervised curve quality with
    paired supervision from the LOL dataset.
    """

    def __init__(self, config):
        super().__init__()
        self.spatial_loss = SpatialConsistencyLoss()
        self.exposure_loss = ExposureControlLoss(target_mean=config.EXPOSURE_MEAN)
        self.color_loss = ColorConstancyLoss()
        self.illumination_loss = IlluminationSmoothnessLoss()
        self.ssim_loss = SSIMLoss()
        self.l1_loss = nn.L1Loss()

        # Only load VGG if perceptual weight > 0 (saves ~500MB RAM on CPU)
        if config.W_PERCEPTUAL > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

        # Loss weights from config
        self.w_spa = config.W_SPATIAL
        self.w_exp = config.W_EXPOSURE
        self.w_col = config.W_COLOR
        self.w_ill = config.W_ILLUMINATION
        self.w_rec = config.W_RECONSTRUCTION
        self.w_per = config.W_PERCEPTUAL
        self.w_ssim = config.W_SSIM

    def forward(self, enhanced, original, curve_params, target=None):
        """
        Compute combined loss.

        Args:
            enhanced: Enhanced image [B, 3, H, W].
            original: Input low-light image [B, 3, H, W].
            curve_params: Curve parameters from the network.
            target: Ground truth normal-light image (optional, for supervised training).

        Returns:
            total_loss: Weighted sum of all losses.
            loss_dict: Dictionary of individual loss values.
        """
        # Zero-reference losses
        L_spa = self.spatial_loss(enhanced, original)
        L_exp = self.exposure_loss(enhanced)
        L_col = self.color_loss(enhanced)
        L_ill = self.illumination_loss(curve_params)

        total_loss = (self.w_spa * L_spa +
                      self.w_exp * L_exp +
                      self.w_col * L_col +
                      self.w_ill * L_ill)

        loss_dict = {
            'spatial': L_spa.item(),
            'exposure': L_exp.item(),
            'color': L_col.item(),
            'illumination': L_ill.item(),
        }

        # Supervised losses (when paired ground truth is available)
        if target is not None:
            L_rec = self.l1_loss(enhanced, target)
            L_ssim = self.ssim_loss(enhanced, target)

            total_loss += self.w_rec * L_rec + self.w_ssim * L_ssim

            loss_dict.update({
                'reconstruction': L_rec.item(),
                'ssim': L_ssim.item(),
            })

            if self.perceptual_loss is not None:
                L_per = self.perceptual_loss(enhanced, target)
                total_loss += self.w_per * L_per
                loss_dict['perceptual'] = L_per.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
