import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_gaussian_kernel2d(kernel_size=5, sigma=1.5):
    """Generates a 2D Gaussian kernel tensor."""
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    gaussian1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gaussian1d = gaussian1d / gaussian1d.sum()
    gaussian2d = torch.outer(gaussian1d, gaussian1d)
    return gaussian2d.unsqueeze(0).unsqueeze(0) # (1, 1, K, K)

class DegradationConsistencyLoss(nn.Module):
    """
    Penalizes deviations when the super-resolved output is degraded back to
    low-resolution (using the known Gaussian PSF and 2x downsampling).
    """
    def __init__(self, kernel_size=5, sigma=1.5):
        super().__init__()
        kernel = get_gaussian_kernel2d(kernel_size, sigma)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_size // 2

    def forward(self, sr_img, lr_img):
        # Apply Gaussian PSF blur
        blurred = F.conv2d(sr_img, self.kernel, padding=self.padding)
        # 2x Downsample via average pooling
        degraded = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        # Compare to input low-resolution TIR
        return F.l1_loss(degraded, lr_img)

class EdgeGradientLoss(nn.Module):
    """
    Edge-gradient penalty using Sobel filters to preserve boundary structures.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        target_dx = F.conv2d(target, self.sobel_x, padding=1)
        target_dy = F.conv2d(target, self.sobel_y, padding=1)
        
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        return loss_x + loss_y

# ----------------------------------------------------
# Stage 2: Discretized Logistic Mixture Loss
# ----------------------------------------------------

class DiscretizedLogisticMixtureNLLLoss(nn.Module):
    """
    Negative Log-Likelihood (NLL) Loss for a Discretized Logistic Mixture distribution over RGB values.
    Uses log-sum-exp and log1p/expm1 for numerical stability.
    """
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon # Variance scale floor

    def forward(self, logit_weights, means, log_scales, targets, pixel_weights=None):
        """
        Args:
            logit_weights: (B, K, H, W) - mixing weights before softmax
            means: (B, K, 3, H, W) - component means for R, G, B
            log_scales: (B, K, 3, H, W) - component log-scales
            targets: (B, 3, H, W) - target RGB values (normalized/rescaled to [0, 255])
            pixel_weights: optional (B, H, W) per-pixel loss weights (colour-
                rarity rebalancing, Zhang et al. style: rare colours such as
                water-blue / snow-white are up-weighted so the plain mean NLL
                cannot average them away in favour of the dominant land palette)

        Gradient-safe formulation (PixelCNN++ style). The naive
        log(clamp(cdf_plus - cdf_min, 1e-7)) zeroes the gradient whenever the
        target sits more than a few scales from a component mean (the clamp
        floor eats the whole expression), which makes recovery from a bad mean
        impossible. Here the saturated regime falls back to the logistic
        log-PDF (x bin width 1), whose gradient w.r.t. the mean never vanishes,
        and the edge bins use exact -softplus forms instead of clamped logs.
        """
        B, K, C, H, W = means.shape
        targets = targets.unsqueeze(1) # (B, 1, 3, H, W)

        # Softmax mixing weights in log-space
        log_weights = F.log_softmax(logit_weights, dim=1) # (B, K, H, W)

        # Enforce scale floor
        scales = F.softplus(log_scales) + self.epsilon # (B, K, 3, H, W)
        inv_s = 1.0 / scales
        centered = targets - means

        plus_in = inv_s * (centered + 0.5)
        minus_in = inv_s * (centered - 0.5)

        # Interior bin probability and its gradient-safe fallback
        cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(minus_in)
        mid_in = inv_s * centered
        # log logistic-pdf at the bin centre (+ log bin-width 1): stays informative
        # (and differentiable w.r.t. means) even when the CDF difference underflows.
        log_pdf_mid = mid_in - torch.log(scales) - 2.0 * F.softplus(mid_in)
        log_probs_interior = torch.where(
            cdf_delta > 1e-5,
            torch.log(torch.clamp(cdf_delta, min=1e-10)),
            log_pdf_mid,
        )

        # Edge bins, numerically exact: log sigmoid(x) = x - softplus(x),
        # log(1 - sigmoid(x)) = -softplus(x). No clamps, no dead gradients.
        log_probs_left = plus_in - F.softplus(plus_in)   # target = 0
        log_probs_right = -F.softplus(minus_in)          # target = 255

        # Select probabilities based on targets
        log_probs = torch.where(targets < 0.001, log_probs_left,
                                torch.where(targets > 254.999, log_probs_right, log_probs_interior))

        # Sum probabilities over RGB channels
        log_probs_rgb = log_probs.sum(dim=2) # (B, K, H, W)

        # Add mixing weights: log(pi_k * p(RGB|k)) = log(pi_k) + log(p(RGB|k))
        log_joint = log_weights + log_probs_rgb # (B, K, H, W)

        # Log-sum-exp over the K mixture components
        nll = -torch.logsumexp(log_joint, dim=1) # (B, H, W)

        if pixel_weights is not None:
            return (nll * pixel_weights).sum() / torch.clamp(pixel_weights.sum(), min=1.0)
        return nll.mean()
