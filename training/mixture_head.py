import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureHead(nn.Module):
    """
    Discretized Logistic Mixture Head for Color Synthesis.
    Outputs K components of mixing weights (logits), RGB means, and log-scales (7K total channels).
    Conditioned on upsampled backbone features and the super-resolved TIR image.
    """
    def __init__(self, in_features=32, K=6):
        super().__init__()
        self.K = K
        
        # Convolutions to refine upsampled features
        # Input features channel + 1 channel from SR-TIR
        self.conv1 = nn.Conv2d(in_features + 1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Output projection layer (7K channels: K weights, 3K means, 3K scales)
        self.proj = nn.Conv2d(64, 7 * K, kernel_size=1)

    def forward(self, features, sr_tir):
        # features['f1'] has shape (B, in_features, H, W) where H, W = 256
        # sr_tir has shape (B, 1, 2H, 2W) where 2H, 2W = 512
        
        # Upsample backbone features to match SR-TIR spatial resolution (512x512)
        f1_upsampled = F.interpolate(features["f1"], size=sr_tir.shape[-2:], mode="bilinear", align_corners=False)
        
        # Concatenate features and SR-TIR along channel dimension
        x = torch.cat([f1_upsampled, sr_tir], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Shape: (B, 7 * K, 512, 512)
        out = self.proj(x)
        
        # Split outputs into logits, means, and log-scales
        # Logits (weights): (B, K, 512, 512)
        # Means: (B, K, 3, 512, 512)
        # Log-scales: (B, K, 3, 512, 512)
        logit_weights = out[:, :self.K, ...]
        
        # Reshape helpers
        B, _, H, W = out.shape
        means = out[:, self.K : 4*self.K, ...].view(B, self.K, 3, H, W)
        log_scales = out[:, 4*self.K :, ...].view(B, self.K, 3, H, W)
        
        return logit_weights, means, log_scales
