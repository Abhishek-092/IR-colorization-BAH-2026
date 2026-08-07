import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excite channel attention (RCAN style)."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class RCAB(nn.Module):
    """Residual Channel-Attention Block (RCAB)."""
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            ChannelAttention(ch),
        )

    def forward(self, x):
        return x + self.body(x)


class SRHead(nn.Module):
    """
    Multi-scale thermal Super-Resolution head (~0.6M parameters).
    Upsamples 200m backbone feature maps to 100m brightness temperature (2x).

    Fuses multi-scale backbone feature maps (f1-f4) with residual channel-attention
    blocks before and after sub-pixel convolution upsampling. Includes a bilinear
    global-residual path to learn high-frequency corrections relative to bilinear input.
    """
    def __init__(self, in_channels=32, out_channels=1, ch=64):
        super().__init__()
        # Project deeper context features to a common width
        self.proj_f2 = nn.Conv2d(in_channels * 2, in_channels, 1)   # f2: 64
        self.proj_f3 = nn.Conv2d(in_channels * 4, in_channels, 1)   # f3: 128
        self.proj_f4 = nn.Conv2d(in_channels * 8, in_channels, 1)   # f4: 256

        # Fuse f1 + 3 context maps + low-res input at f1 resolution
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4 + 1, ch, 3, padding=1), nn.ReLU(inplace=True))

        # Low-resolution refinement trunk
        self.body_lr = nn.Sequential(RCAB(ch), RCAB(ch))

        # 2x sub-pixel upsample
        self.upsample = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True))

        # High-resolution refinement trunk
        self.body_hr = nn.Sequential(RCAB(ch), RCAB(ch))

        # Reconstruction to a single brightness-temperature channel.
        self.recon = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, out_channels, 3, padding=1))

    def forward(self, features, lr_tir=None):
        f1 = features["f1"]
        size = f1.shape[-2:]

        def up(x):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        f2 = up(self.proj_f2(features["f2"]))
        f3 = up(self.proj_f3(features["f3"]))
        f4 = up(self.proj_f4(features["f4"]))

        # Include the raw LR input directly so the fusion sees the true signal.
        lr_at_f1 = lr_tir if (lr_tir is not None and lr_tir.shape[-2:] == size) \
            else (F.interpolate(lr_tir, size=size, mode="bilinear", align_corners=False)
                  if lr_tir is not None else torch.zeros_like(f1[:, :1]))

        x = torch.cat([f1, f2, f3, f4, lr_at_f1], dim=1)
        x = self.fuse(x)
        x = x + self.body_lr(x)
        x = self.upsample(x)              # -> (B, ch, 2H, 2W)
        x = x + self.body_hr(x)
        x = self.recon(x)                 # -> (B, 1, 2H, 2W)

        # Global residual: learn the correction on top of a clean bilinear upscale.
        if lr_tir is not None:
            x = x + F.interpolate(lr_tir, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x
