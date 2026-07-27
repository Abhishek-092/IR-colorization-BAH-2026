import torch
import pytest
from training.loss_functions import DiscretizedLogisticMixtureNLLLoss

def test_discretized_logistic_mixture_stability():
    """
    Asserts that the loss function is numerically stable under degenerate scales.
    """
    B, K, C, H, W = 2, 6, 3, 32, 32
    logit_weights = torch.randn(B, K, H, W)
    means = torch.randn(B, K, C, H, W) * 128.0 + 127.0
    
    # Intentionally tiny scale parameters (will trigger division/scale issues if un-stabilized)
    log_scales = torch.ones(B, K, C, H, W) * -20.0 
    
    targets = torch.randint(0, 256, (B, C, H, W)).float()

    # Epsilon = 1.0 floor
    nll_loss_fn = DiscretizedLogisticMixtureNLLLoss(epsilon=1.0)
    loss = nll_loss_fn(logit_weights, means, log_scales, targets)

    # Check for valid numerical output
    assert not torch.isnan(loss).any(), "Loss contains NaNs under near-degenerate scales"
    assert not torch.isinf(loss).any(), "Loss contains Infs under near-degenerate scales"
    assert loss.item() > 0, "Loss value is zero or negative, indicating probability leakage"
    print(f"Stable loss verified successfully: {loss.item():.4f}")

def test_extreme_large_scales_and_out_of_bounds_targets():
    """
    Asserts that the loss function is numerically stable under extremely large scales and out of bound targets.
    """
    B, K, C, H, W = 2, 6, 3, 32, 32
    logit_weights = torch.randn(B, K, H, W)
    means = torch.randn(B, K, C, H, W) * 128.0 + 127.0
    
    # Intentionally large scale parameters (tests the log1p clamp boundary)
    log_scales = torch.ones(B, K, C, H, W) * 30.0 
    
    # Targets out of bounds
    targets = torch.randn(B, C, H, W) * 50.0 + 127.0
    targets[0, 0, 0, 0] = -5.0
    targets[1, 1, 1, 1] = 260.0

    nll_loss_fn = DiscretizedLogisticMixtureNLLLoss(epsilon=1.0)
    loss = nll_loss_fn(logit_weights, means, log_scales, targets)

    assert not torch.isnan(loss).any(), "Loss contains NaNs under extreme scales/targets"
    assert not torch.isinf(loss).any(), "Loss contains Infs under extreme scales/targets"
    assert loss.item() > 0, "Loss value is zero or negative"

