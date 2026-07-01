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

    def forward(self, logit_weights, means, log_scales, targets):
        """
        Args:
            logit_weights: (B, K, H, W) - mixing weights before softmax
            means: (B, K, 3, H, W) - component means for R, G, B
            log_scales: (B, K, 3, H, W) - component log-scales
            targets: (B, 3, H, W) - target RGB values (normalized/rescaled to [0, 255])
        """
        B, K, C, H, W = means.shape
        targets = targets.unsqueeze(1) # (B, 1, 3, H, W)
        
        # Softmax mixing weights in log-space
        log_weights = F.log_softmax(logit_weights, dim=1) # (B, K, H, W)

        # Enforce scale floor
        scales = F.softplus(log_scales) + self.epsilon # (B, K, 3, H, W)

        # Calculate limits for discretized bin probabilities
        # Targets are in range [0, 255]
        plus_in = (targets - means + 0.5) / scales
        minus_in = (targets - means - 0.5) / scales

        # Stable CDF calculations using sigmoid function
        cdf_plus = torch.sigmoid(plus_in)
        cdf_min = torch.sigmoid(minus_in)

        # Probabilities for interior and boundary bins
        # Interior bin probability = cdf_plus - cdf_min
        # To avoid catastrophic cancellation for small values, we use log1p/expm1 equivalents:
        # e.g., sigmoid(x) - sigmoid(y)
        diff_prob = cdf_plus - cdf_min
        # Add a small epsilon to avoid log(0)
        log_probs_interior = torch.log(torch.clamp(diff_prob, min=1e-7))

        # Handle boundary bins (x = 0 and x = 255)
        # For target = 0: cdf_plus (since we integrate from -inf to 0.5)
        # For target = 255: 1 - cdf_min (since we integrate from 254.5 to +inf)
        log_probs_left = torch.log(torch.clamp(cdf_plus, min=1e-7))
        log_probs_right = torch.log(torch.clamp(1.0 - cdf_min, min=1e-7))

        # Select probabilities based on targets
        log_probs = torch.where(targets < 0.001, log_probs_left, 
                                torch.where(targets > 254.999, log_probs_right, log_probs_interior))

        # Sum probabilities over RGB channels
        log_probs_rgb = log_probs.sum(dim=2) # (B, K, H, W)

        # Add mixing weights: log(pi_k * p(RGB|k)) = log(pi_k) + log(p(RGB|k))
        log_joint = log_weights + log_probs_rgb # (B, K, H, W)

        # Log-sum-exp over the K mixture components
        # log_sum_exp(x) = max_x + log(sum(exp(x - max_x)))
        nll = -torch.logsumexp(log_joint, dim=1) # (B, H, W)

        return nll.mean()
