import torch
import torch.nn as nn
import torch.nn.functional as F

class SRHead(nn.Module):
    """
    Deterministic Super-Resolution Head (~0.3M parameters).
    Upsamples 200m input features to 100m spatial resolution (2x upsampling).
    """
    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()
        
        # Refinement convolution on backbone's highest resolution feature map (f1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 2x Upsampling using PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),  # 16 channels -> 4 channels upsampled 2x
            nn.ReLU(inplace=True)
        )

        # Final projection to 1 channel (Brightness Temperature)
        self.conv_out = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            self.relu,
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, features, lr_tir=None):
        # We use the f1 feature map which has same spatial dimensions as input (200m, e.g., 256x256)
        x = features["f1"]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.upsample(x) # Upsampled to (B, 4, 2H, 2W)
        x = self.conv_out(x) # Out: (B, 1, 2H, 2W)
        
        if lr_tir is not None:
            lr_upsampled = F.interpolate(lr_tir, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + lr_upsampled
            
        return x
