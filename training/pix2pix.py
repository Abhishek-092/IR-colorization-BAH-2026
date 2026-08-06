import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FeatureUNetGenerator(nn.Module):
    """
    Feature-Conditioned UNet Generator for Thermal-to-RGB translation.
    Inputs:
        - sr_tir: (B, 1, H, W) Super-Resolved Thermal field (e.g. 512x512)
        - features: dict {"f1": (B,32,H',W'), "f2": (B,64,H'/2,W'/2),
                          "f3": (B,128,H'/4,W'/4), "f4": (B,256,H'/8,W'/8)}
                    Multi-scale backbone features (same dict that SRHead/MixtureHead use).
    Outputs:
        - (B, 3, H, W) RGB predictions in [0, 255]
    """
    def __init__(self, in_channels=1, out_channels=3, feat_channels=64, base_channels=None):
        super().__init__()
        base_channels = base_channels if base_channels is not None else feat_channels
        # UNet encoder
        self.inc = DoubleConv(in_channels, base_channels)  # 64
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))  # 128
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))  # 256
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))  # 512

        # Multi-scale backbone feature projections (matches ResNetBackbone output)
        # All four levels are projected to a common width and fused at the
        # bottleneck, giving the generator the same multi-scale context that
        # MixtureHead enjoys — essential for distinguishing materials at similar
        # temperatures (e.g. river vs. wet soil).
        ctx = 32  # common projection width per feature level
        self.proj_f1 = nn.Conv2d(32, ctx, kernel_size=1)    # f1: 32ch
        self.proj_f2 = nn.Conv2d(64, ctx, kernel_size=1)    # f2: 64ch
        self.proj_f3 = nn.Conv2d(128, ctx, kernel_size=1)   # f3: 128ch
        self.proj_f4 = nn.Conv2d(256, ctx, kernel_size=1)   # f4: 256ch

        # Bottleneck: UNet x4 (512) + 4 projected feature maps (4*32=128) = 640
        bottleneck_in = base_channels * 8 + ctx * 4
        self.bottleneck = DoubleConv(bottleneck_in, base_channels * 8)

        # UNet decoder (skip connections from encoder)
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, sr_tir, features=None):
        # --- UNet Encoder ---
        x1 = self.inc(sr_tir)          # (B, 64, H, W)
        x2 = self.down1(x1)            # (B, 128, H/2, W/2)
        x3 = self.down2(x2)            # (B, 256, H/4, W/4)
        x4 = self.down3(x3)            # (B, 512, H/8, W/8)

        # --- Inject multi-scale backbone features at the bottleneck ---
        if features is not None and isinstance(features, dict):
            bn_size = x4.shape[-2:]

            def to_bn(t):
                if t.shape[-2:] != bn_size:
                    return F.interpolate(t, size=bn_size, mode='bilinear', align_corners=False)
                return t

            pf1 = to_bn(self.proj_f1(features["f1"]))
            pf2 = to_bn(self.proj_f2(features["f2"]))
            pf3 = to_bn(self.proj_f3(features["f3"]))
            pf4 = to_bn(self.proj_f4(features["f4"]))
            bn_in = torch.cat([x4, pf1, pf2, pf3, pf4], dim=1)
        elif features is not None and torch.is_tensor(features):
            bn_size = x4.shape[-2:]
            if features.shape[1] == 64:
                pf2 = self.proj_f2(features)
            else:
                pf2 = features
                if pf2.shape[1] != 32:
                    pf2 = nn.Conv2d(pf2.shape[1], 32, 1, device=features.device)(pf2)
            if pf2.shape[-2:] != bn_size:
                pf2 = F.interpolate(pf2, size=bn_size, mode='bilinear', align_corners=False)
            zeros = torch.zeros(x4.shape[0], 96, bn_size[0], bn_size[1], device=x4.device, dtype=x4.dtype)
            bn_in = torch.cat([x4, pf2, zeros], dim=1)
        else:
            # No backbone features: zero-pad to match expected bottleneck width
            B, _, H, W = x4.shape
            zeros = torch.zeros(B, 128, H, W, device=x4.device, dtype=x4.dtype)
            bn_in = torch.cat([x4, zeros], dim=1)

        b = self.bottleneck(bn_in)      # (B, 512, H/8, W/8)

        # --- UNet Decoder (skip connections) ---
        u1 = self.up1(b)                # (B, 256, H/4, W/4)
        u1 = torch.cat([u1, x3], dim=1)
        x = self.conv_up1(u1)

        u2 = self.up2(x)                # (B, 128, H/2, W/2)
        u2 = torch.cat([u2, x2], dim=1)
        x = self.conv_up2(u2)

        u3 = self.up3(x)                # (B, 64, H, W)
        u3 = torch.cat([u3, x1], dim=1)
        x = self.conv_up3(u3)

        out = torch.sigmoid(self.outc(x)) * 255.0
        return out


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator with Spectral Normalization for 1-Lipschitz stability.
    Inputs:
        - Thermal + RGB concatenated tensor: (B, 4, H, W)
    Outputs:
        - (B, 1, H/8, W/8) patch validity predictions
    """
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        def conv_sn(in_c, out_c, stride=2):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)
            )

        self.model = nn.Sequential(
            conv_sn(in_channels, base_channels, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            conv_sn(base_channels, base_channels * 2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            conv_sn(base_channels * 2, base_channels * 4, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            conv_sn(base_channels * 4, base_channels * 8, stride=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            conv_sn(base_channels * 8, 1, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class SpatialColorLoss(nn.Module):
    """
    Pure PyTorch, zero-dependency offline loss for thermal-to-RGB translation.
    Combines:
    1. L1 loss (pixel fidelity)
    2. Sobel edge loss (spatial structure alignment)
    3. Color channel ratio loss (prevents channel collapse)
    """
    def __init__(self, lambda_l1=100.0, lambda_grad=10.0, lambda_color=5.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_color = lambda_color

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _sobel_edges(self, img):
        if img.shape[1] == 3:
            lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            lum = img
        grad_x = F.conv2d(lum, self.sobel_x, padding=1)
        grad_y = F.conv2d(lum, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, pred_rgb, target_rgb):
        loss_l1 = F.l1_loss(pred_rgb, target_rgb)

        pred_edge = self._sobel_edges(pred_rgb)
        target_edge = self._sobel_edges(target_rgb)
        loss_grad = F.l1_loss(pred_edge, target_edge)

        pred_channel_means = pred_rgb.mean(dim=(2, 3))
        target_channel_means = target_rgb.mean(dim=(2, 3))
        loss_color = F.l1_loss(pred_channel_means, target_channel_means)

        total_loss = (self.lambda_l1 * loss_l1) + (self.lambda_grad * loss_grad) + (self.lambda_color * loss_color)
        return total_loss, loss_l1, loss_grad, loss_color
