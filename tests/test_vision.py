import torch
import torch.nn as nn
import pytest

from nn.vision.model_getter import get_resnet


def test_get_resnet18_is_nn_module():
    model = get_resnet("resnet18", weights=None)
    assert isinstance(model, nn.Module)


def test_get_resnet18_fc_is_identity():
    model = get_resnet("resnet18", weights=None)
    assert isinstance(model.fc, nn.Identity)


def test_get_resnet18_forward_shape():
    model = get_resnet("resnet18", weights=None)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
    assert out.shape == (2, 512)
