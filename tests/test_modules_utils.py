import math
import torch
import pytest

from nn.modules.utils import trunc_normal_init_, cosine_schedule_with_warmup_lr_lambda, stablemax_cross_entropy


def test_trunc_normal_init_shape():
    t = torch.empty(4, 8)
    out = trunc_normal_init_(t, std=0.02)
    assert out.shape == (4, 8)


def test_trunc_normal_init_bounded():
    t = torch.empty(1000, 64)
    trunc_normal_init_(t, std=1.0, lower=-2.0, upper=2.0)
    # Values should be within [lower * comp_std, upper * comp_std]; check they are finite and bounded
    assert torch.isfinite(t).all()
    # Rough sanity: most values within a few std of 0
    assert t.abs().max().item() < 10.0


def test_trunc_normal_init_zero_std():
    t = torch.ones(4, 4)
    trunc_normal_init_(t, std=0)
    assert torch.all(t == 0)


def test_cosine_schedule_warmup_linear():
    base_lr = 1e-3
    warmup_steps = 100
    total_steps = 1000

    # During warmup, lr should be linear from 0 to base_lr
    lr_0 = cosine_schedule_with_warmup_lr_lambda(
        0, base_lr=base_lr, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    lr_50 = cosine_schedule_with_warmup_lr_lambda(
        50, base_lr=base_lr, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    lr_100 = cosine_schedule_with_warmup_lr_lambda(
        100, base_lr=base_lr, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    assert lr_0 == 0.0
    assert abs(lr_50 - base_lr * 50 / 100) < 1e-9
    assert abs(lr_100 - base_lr) < 1e-9


def test_cosine_schedule_post_warmup_cosine_decay():
    base_lr = 1.0
    warmup_steps = 0
    total_steps = 100
    min_ratio = 0.0

    lr_0 = cosine_schedule_with_warmup_lr_lambda(
        0, base_lr=base_lr, num_warmup_steps=warmup_steps, num_training_steps=total_steps, min_ratio=min_ratio
    )
    lr_end = cosine_schedule_with_warmup_lr_lambda(
        total_steps, base_lr=base_lr, num_warmup_steps=warmup_steps, num_training_steps=total_steps, min_ratio=min_ratio
    )
    assert abs(lr_0 - base_lr) < 1e-9
    # At end, cosine should approach 0 (min_ratio=0)
    assert lr_end < 0.01


def test_cosine_schedule_respects_min_ratio():
    base_lr = 1.0
    min_ratio = 0.1
    total_steps = 1000
    warmup_steps = 0
    lr_end = cosine_schedule_with_warmup_lr_lambda(
        total_steps, base_lr=base_lr, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps, min_ratio=min_ratio
    )
    assert lr_end >= base_lr * min_ratio - 1e-9


def test_stablemax_cross_entropy_shape():
    logits = torch.randn(4, 10)
    labels = torch.randint(1, 10, (4,))  # avoid ignore_index=0
    losses = stablemax_cross_entropy(logits, labels, ignore_index=0)
    assert losses.shape == (4,)


def test_stablemax_cross_entropy_ignore_index():
    logits = torch.randn(4, 10)
    labels = torch.zeros(4, dtype=torch.long)  # all ignored
    losses = stablemax_cross_entropy(logits, labels, ignore_index=0)
    assert (losses == 0).all()


def test_stablemax_cross_entropy_correct_class_lower():
    # Strong logit for class 1
    logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]], dtype=torch.float32)
    label_correct = torch.tensor([1])
    label_wrong = torch.tensor([2])
    loss_correct = stablemax_cross_entropy(logits, label_correct, ignore_index=-1)
    loss_wrong = stablemax_cross_entropy(logits, label_wrong, ignore_index=-1)
    assert loss_correct.item() < loss_wrong.item()
