import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard ResNet bottleneck or basic block for feature extraction.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNetBackbone(nn.Module):
    """
    Shared ResNet-style encoder backbone (~2.5M parameters).
    Sized specifically for the low information density of single-band thermal inputs.
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = base_channels
        
        # Initial projection
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # 4 ResNet Stages with channel widths [32, 64, 128, 256]
        self.stage1 = self._make_layer(base_channels, 2, stride=1)
        self.stage2 = self._make_layer(base_channels * 2, 2, stride=2)
        self.stage3 = self._make_layer(base_channels * 4, 2, stride=2)
        self.stage4 = self._make_layer(base_channels * 8, 2, stride=2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (B, 1, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        f1 = self.stage1(x)  # (B, 32, H, W)
        f2 = self.stage2(f1) # (B, 64, H/2, W/2)
        f3 = self.stage3(f2) # (B, 128, H/4, W/4)
        f4 = self.stage4(f3) # (B, 256, H/8, W/8)

        # Return dict of multi-scale features
        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4
        }
