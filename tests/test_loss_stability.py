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
