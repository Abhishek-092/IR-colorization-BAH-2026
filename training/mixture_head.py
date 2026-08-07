import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureHead(nn.Module):
    """
    Discretized Logistic Mixture Head for Color Synthesis.
    Outputs K components of mixing weights (logits), RGB means, and log-scales (7K channels total).

    Conditioned on super-resolved thermal input and multi-scale backbone features (f1..f4).
    Deep features (f2..f4) provide spatial context to resolve cross-spectral material ambiguity.
    """
    def __init__(self, in_features=32, K=6):
        super().__init__()
        self.K = K

        # Project deeper context feature maps to a smaller common width
        ctx = 24
        self.proj_f2 = nn.Conv2d(in_features * 2, ctx, kernel_size=1)   # f2: 64ch
        self.proj_f3 = nn.Conv2d(in_features * 4, ctx, kernel_size=1)   # f3: 128ch
        self.proj_f4 = nn.Conv2d(in_features * 8, ctx, kernel_size=1)   # f4: 256ch

        # Fused input: f1 + projected context maps + SR-TIR + thermal texture channels
        # (local std dev and gradient magnitude derived from SR-TIR)
        fused_ch = in_features + 3 * ctx + 1 + 2
        self.conv1 = nn.Conv2d(fused_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Output projection layer (7K channels: K weights, 3K means, 3K scales)
        self.proj = nn.Conv2d(64, 7 * K, kernel_size=1)

    @staticmethod
    def _thermal_texture(sr_tir):
        """Two texture channels from the SR thermal field (differentiable,
        parameter-free): local standard deviation over an 11x11 window
        (smoothness — near zero over water) and gradient magnitude
        (edge/roughness density — high over urban / broken terrain).
        Both lightly scaled so they land in a similar range to the BT input."""
        mean = F.avg_pool2d(sr_tir, 11, stride=1, padding=5)
        mean_sq = F.avg_pool2d(sr_tir * sr_tir, 11, stride=1, padding=5)
        local_std = torch.sqrt(torch.clamp(mean_sq - mean * mean, min=0.0))
        gx = sr_tir[..., :, 1:] - sr_tir[..., :, :-1]
        gy = sr_tir[..., 1:, :] - sr_tir[..., :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return torch.cat([local_std * 10.0, grad * 10.0], dim=1)

    def forward(self, features, sr_tir):
        # sr_tir has shape (B, 1, 2H, 2W); fuse everything at that resolution.
        size = sr_tir.shape[-2:]

        def up(x):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        f1 = up(features["f1"])
        f2 = up(self.proj_f2(features["f2"]))
        f3 = up(self.proj_f3(features["f3"]))
        f4 = up(self.proj_f4(features["f4"]))

        # Concatenate multi-scale context + super-resolved thermal + texture
        tex = self._thermal_texture(sr_tir)
        x = torch.cat([f1, f2, f3, f4, sr_tir, tex], dim=1)

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

        B, _, H, W = out.shape
        means = out[:, self.K : 4*self.K, ...].view(B, self.K, 3, H, W)
        log_scales = out[:, 4*self.K :, ...].view(B, self.K, 3, H, W)

        return logit_weights, means, log_scales
