import torch
import pytest
from training.pix2pix import FeatureUNetGenerator, PatchGANDiscriminator, SpatialColorLoss


def test_feature_unet_generator_shapes():
    gen = FeatureUNetGenerator(in_channels=1, out_channels=3, feat_channels=64)
    gen.eval()

    sr_tir = torch.randn(2, 1, 512, 512)
    features_dict = {
        "f1": torch.randn(2, 32, 512, 512),
        "f2": torch.randn(2, 64, 256, 256),
        "f3": torch.randn(2, 128, 128, 128),
        "f4": torch.randn(2, 256, 64, 64),
    }

    with torch.no_grad():
        rgb_out = gen(sr_tir, features_dict)

    assert rgb_out.shape == (2, 3, 512, 512), f"Expected (2, 3, 512, 512), got {rgb_out.shape}"
    assert (rgb_out >= 0.0).all() and (rgb_out <= 255.0).all(), "RGB output out of range [0, 255]"

    # Test tensor fallback
    features_tensor = torch.randn(2, 64, 256, 256)
    with torch.no_grad():
        rgb_out_tensor = gen(sr_tir, features_tensor)
    assert rgb_out_tensor.shape == (2, 3, 512, 512)

    # Test None features
    with torch.no_grad():
        rgb_out_none = gen(sr_tir, None)
    assert rgb_out_none.shape == (2, 3, 512, 512)


def test_patchgan_discriminator_shapes():
    disc = PatchGANDiscriminator(in_channels=4)
    disc.eval()

    input_pair = torch.randn(2, 4, 512, 512)

    with torch.no_grad():
        out = disc(input_pair)

    assert out.ndim == 4 and out.shape[0] == 2 and out.shape[1] == 1, f"Unexpected discriminator shape {out.shape}"


def test_spatial_color_loss_backward():
    loss_fn = SpatialColorLoss()
    pred_rgb = torch.randn(2, 3, 256, 256, requires_grad=True)
    target_rgb = torch.randn(2, 3, 256, 256)

    total_loss, l1, grad, color = loss_fn(pred_rgb, target_rgb)
    total_loss.backward()

    assert pred_rgb.grad is not None, "Gradient was not propagated to pred_rgb"
    assert not torch.isnan(pred_rgb.grad).any(), "NaN found in gradients"
    assert total_loss.item() > 0, "Loss should be strictly positive"
