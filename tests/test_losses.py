import torch
import pytest

from nn.losses.kl_divergence import kl_divergence
from nn.losses.stable_max_loss import StableMaxCrossEntropyLoss


# --- kl_divergence ---

def test_kl_divergence_return_shapes():
    B, D = 4, 8
    mu = torch.randn(B, D)
    logvar = torch.randn(B, D)
    total, dim_wise, mean = kl_divergence(mu, logvar)
    assert total.shape == (1,)
    assert dim_wise.shape == (D,)
    assert mean.shape == (1,)


def test_kl_divergence_zero_at_prior():
    B, D = 8, 16
    mu = torch.zeros(B, D)
    logvar = torch.zeros(B, D)
    total, _, mean = kl_divergence(mu, logvar)
    assert torch.allclose(total, torch.zeros(1), atol=1e-6)
    assert torch.allclose(mean, torch.zeros(1), atol=1e-6)


def test_kl_divergence_positive_for_nonzero():
    B, D = 4, 8
    mu = torch.ones(B, D) * 2.0
    logvar = torch.ones(B, D) * 1.0
    total, _, _ = kl_divergence(mu, logvar)
    assert total.item() > 0.0


# --- StableMaxCrossEntropyLoss ---

def test_stable_max_loss_basic():
    loss_fn = StableMaxCrossEntropyLoss()
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0.0


def test_stable_max_loss_ignore_index():
    loss_fn = StableMaxCrossEntropyLoss(ignore_index=0)
    logits = torch.randn(8, 10)
    targets = torch.zeros(8, dtype=torch.long)  # all ignored
    # Should return 0 loss since all tokens are ignored
    loss = loss_fn(logits, targets)
    assert loss.item() == 0.0


def test_stable_max_loss_no_nan_extreme_logits():
    loss_fn = StableMaxCrossEntropyLoss()
    logits = torch.tensor([[1e4, -1e4, 0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0])
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_stable_max_loss_correct_class_lower():
    loss_fn = StableMaxCrossEntropyLoss()
    # Give strong signal to class 0
    logits_good = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    logits_random = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0])
    loss_good = loss_fn(logits_good, targets)
    loss_random = loss_fn(logits_random, targets)
    assert loss_good.item() < loss_random.item()
