import numpy as np
import torch
import pytest

from nn.modules.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def test_single_field_limits_last_n_dims2():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    norm = SingleFieldLinearNormalizer()
    norm.fit(data, mode="limits", last_n_dims=2)
    datan = norm.normalize(data)

    assert datan.shape == data.shape
    assert np.allclose(datan.max().item(), 1.0)
    assert np.allclose(datan.min().item(), -1.0)

    dataun = norm.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


def test_single_field_limits_no_offset():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    norm = SingleFieldLinearNormalizer()
    norm.fit(data, mode="limits", last_n_dims=1, fit_offset=False)
    datan = norm.normalize(data)

    assert datan.shape == data.shape
    assert np.allclose(datan.max().item(), 1.0, atol=1e-3)
    assert np.allclose(datan.min().item(), 0.0, atol=1e-3)

    dataun = norm.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


def test_single_field_gaussian():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    norm = SingleFieldLinearNormalizer()
    norm.fit(data, mode="gaussian", last_n_dims=0)
    datan = norm.normalize(data)

    assert datan.shape == data.shape
    assert np.allclose(datan.mean().item(), 0.0, atol=1e-3)
    assert np.allclose(datan.std().item(), 1.0, atol=1e-3)

    dataun = norm.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


def test_linear_normalizer_tensor():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    norm = LinearNormalizer()
    norm.fit(data, mode="limits", last_n_dims=2)
    datan = norm.normalize(data)

    assert datan.shape == data.shape
    assert np.allclose(datan.max().item(), 1.0)
    assert np.allclose(datan.min().item(), -1.0)

    dataun = norm.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


def test_linear_normalizer_dict():
    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    norm = LinearNormalizer()
    norm.fit(data)
    datan = norm.normalize(data)
    dataun = norm.unnormalize(datan)

    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)


def test_linear_normalizer_state_dict_roundtrip():
    data = {
        "obs": torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        "action": torch.zeros((1000, 128, 2)).uniform_() * 512,
    }
    norm = LinearNormalizer()
    norm.fit(data)

    state_dict = norm.state_dict()
    norm2 = LinearNormalizer()
    norm2.load_state_dict(state_dict)

    datan = norm2.normalize(data)
    dataun = norm2.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)


def test_create_fit_classmethod():
    data = torch.randn(50, 4)
    norm = SingleFieldLinearNormalizer.create_fit(data, mode="limits", last_n_dims=1)
    datan = norm.normalize(data)
    assert datan.shape == data.shape
    assert torch.allclose(data, norm.unnormalize(datan), atol=1e-6)


def test_create_manual_classmethod():
    scale = torch.tensor([2.0, 0.5], dtype=torch.float32)
    offset = torch.tensor([0.0, 1.0], dtype=torch.float32)
    input_stats = {
        "min": torch.tensor([-1.0, -1.0], dtype=torch.float32),
        "max": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "mean": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "std": torch.tensor([0.5, 2.0], dtype=torch.float32),
    }
    norm = SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    xn = norm.normalize(x)
    assert xn.shape == x.shape


def test_create_identity_classmethod():
    norm = SingleFieldLinearNormalizer.create_identity()
    x = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
    xn = norm.normalize(x)
    assert torch.allclose(x, xn, atol=1e-6)
    assert torch.allclose(x, norm.unnormalize(xn), atol=1e-6)
