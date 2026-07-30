import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureHead(nn.Module):
    """
    Discretized Logistic Mixture Head for Color Synthesis.
    Outputs K components of mixing weights (logits), RGB means, and log-scales (7K total channels).

    Conditioned on the super-resolved TIR plus MULTI-SCALE backbone features
    (f1..f4). Colour is one-to-many from temperature alone: a 300 K pixel can be
    river, wet soil or shadow. The deep features f2..f4 carry the spatial/semantic
    context (texture, shape, neighbourhood) that disambiguates material, so the
    colour output can follow the ground instead of the temperature alone. f2..f4
    are projected to a small common width before fusion to keep the head light.
    """
    def __init__(self, in_features=32, K=6):
        super().__init__()
        self.K = K

        # Project the deeper (context) feature maps to a small common width.
        ctx = 24
        self.proj_f2 = nn.Conv2d(in_features * 2, ctx, kernel_size=1)   # f2: 64ch
        self.proj_f3 = nn.Conv2d(in_features * 4, ctx, kernel_size=1)   # f3: 128ch
        self.proj_f4 = nn.Conv2d(in_features * 8, ctx, kernel_size=1)   # f4: 256ch

        # Fused input: f1 (in_features) + 3 context maps (3*ctx) + SR-TIR (1)
        fused_ch = in_features + 3 * ctx + 1
        self.conv1 = nn.Conv2d(fused_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Output projection layer (7K channels: K weights, 3K means, 3K scales)
        self.proj = nn.Conv2d(64, 7 * K, kernel_size=1)

    def forward(self, features, sr_tir):
        # sr_tir has shape (B, 1, 2H, 2W); fuse everything at that resolution.
        size = sr_tir.shape[-2:]

        def up(x):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        f1 = up(features["f1"])
        f2 = up(self.proj_f2(features["f2"]))
        f3 = up(self.proj_f3(features["f3"]))
        f4 = up(self.proj_f4(features["f4"]))

        # Concatenate multi-scale context + super-resolved thermal
        x = torch.cat([f1, f2, f3, f4, sr_tir], dim=1)

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
