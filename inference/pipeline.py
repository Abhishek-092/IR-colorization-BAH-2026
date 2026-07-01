import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from varna.calibration.planck import dn_to_brightness_temp

class DecodeSubmoduleFP32(nn.Module):
    """
    Decodes raw mixture-head outputs into primary colors and uncertainty maps.
    Strictly forced to execute in FP32 precision to prevent numerical truncation.
    """
    def __init__(self, K=6):
        super().__init__()
        self.K = K
        # Logistic variance constant = pi^2 / 3
        self.register_buffer("logistic_const", torch.tensor(np.pi ** 2 / 3.0, dtype=torch.float32))

    def forward(self, logit_weights, means, log_scales):
        # Ensure FP32 precision
        logit_weights = logit_weights.float()
        means = means.float()
        log_scales = log_scales.float()

        # Compute softmax weights pi_k
        pi = F.softmax(logit_weights, dim=1) # (B, K, H, W)
        pi_unsqueezed = pi.unsqueeze(2) # (B, K, 1, H, W)

        # Scale parameters (softplus + epsilon)
        # Assuming epsilon=1.0 is used for training
        scales = F.softplus(log_scales) + 1.0 # (B, K, 3, H, W)

        # 1. Reported color = mean of the dominant component (argmax of pi)
        k_star = torch.argmax(pi, dim=1, keepdim=True) # (B, 1, H, W)
        # Select means corresponding to k_star
        # Gather along components dimension (dim 1)
        k_star_expanded = k_star.unsqueeze(2).expand(-1, -1, 3, -1, -1) # (B, 1, 3, H, W)
        dominant_color = torch.gather(means, dim=1, index=k_star_expanded).squeeze(1) # (B, 3, H, W)

        # 2. Secondary hypothesis (second highest weight)
        if self.K > 1:
            top_weights, top_indices = torch.topk(pi, k=2, dim=1) # (B, 2, H, W)
            k_sec = top_indices[:, 1:2, ...] # (B, 1, H, W)
            k_sec_expanded = k_sec.unsqueeze(2).expand(-1, -1, 3, -1, -1)
            secondary_color = torch.gather(means, dim=1, index=k_sec_expanded).squeeze(1)
            secondary_weight = top_weights[:, 1, ...] # (B, H, W)
        else:
            secondary_color = dominant_color
            secondary_weight = torch.zeros_like(k_star.squeeze(1))

        # 3. Uncertainty Decomposition (Law of Total Variance)
        # Within-mode variance: sum(pi_k * s_k^2 * pi^2/3)
        within_mode_var = (pi_unsqueezed * (scales ** 2) * self.logistic_const).sum(dim=1) # (B, 3, H, W)

        # Between-mode variance: sum(pi_k * (mu_k - bar_mu)^2)
        # bar_mu (mixture mean) = sum(pi_k * mu_k)
        bar_mu = (pi_unsqueezed * means).sum(dim=1, keepdim=True) # (B, 1, 3, H, W)
        between_mode_var = (pi_unsqueezed * ((means - bar_mu) ** 2)).sum(dim=1) # (B, 3, H, W)

        # Entropy of mixing weights
        # H(pi) = -sum(pi * log(pi))
        log_pi = torch.log(torch.clamp(pi, min=1e-7))
        entropy = -(pi * log_pi).sum(dim=1) # (B, H, W)

        return {
            "dominant_color": dominant_color,
            "secondary_color": secondary_color,
            "secondary_weight": secondary_weight,
            "within_mode_variance": within_mode_var.mean(dim=1),  # Average across RGB channels
            "between_mode_variance": between_mode_var.mean(dim=1),
            "entropy": entropy
        }

class SUTRAMInferencePipeline(nn.Module):
    """
    Fused inference pipeline: Calibration -> Backbone -> SR -> Mixture -> Decode
    Designed to compile into a single execution graph.
    """
    def __init__(self, backbone, sr_head, mixture_head, K=6):
        super().__init__()
        self.backbone = backbone
        self.sr_head = sr_head
        self.mixture_head = mixture_head
        self.decode = DecodeSubmoduleFP32(K=K)

    def forward(self, lr_tir):
        # Input normalization: raw DN -> [0, 1]
        TIR_MIN, TIR_MAX = 20000.0, 35000.0
        lr_tir_norm = torch.clamp((lr_tir - TIR_MIN) / (TIR_MAX - TIR_MIN), 0.0, 1.0)

        features = self.backbone(lr_tir_norm)
        
        # 2. Stage 1 Super-Resolution
        sr_tir = self.sr_head(features, lr_tir_norm)
        
        # 3. Stage 2 Color Parameter Estimation
        logit_weights, means, log_scales = self.mixture_head(features, sr_tir)
        
        # 4. FP32 Decoding
        decode_outputs = self.decode(logit_weights, means, log_scales)
        
        # Denormalize outputs back to raw DN ranges
        sr_tir_dn = torch.clamp(sr_tir * (TIR_MAX - TIR_MIN) + TIR_MIN, TIR_MIN, TIR_MAX)
        # Denormalize dominant and secondary colors to original 0-10000 scale
        RGB_SCALE = 10000.0
        decode_outputs["dominant_color"] = torch.clamp((decode_outputs["dominant_color"] / 255.0) * RGB_SCALE, 0.0, RGB_SCALE)
        decode_outputs["secondary_color"] = torch.clamp((decode_outputs["secondary_color"] / 255.0) * RGB_SCALE, 0.0, RGB_SCALE)
            
        return sr_tir_dn, decode_outputs
