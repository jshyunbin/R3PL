import torch
import pytest

from nn.modules.trm_block import (
    _find_multiple,
    RMSNorm,
    CastedLinear,
    CastedLayerNorm,
    SwiGLU,
    Transformer,
    TransformerDecoder,
    ActionReasoningModule,
)


def test_find_multiple():
    assert _find_multiple(10, 8) % 8 == 0
    assert _find_multiple(10, 8) >= 10
    assert _find_multiple(16, 8) == 16
    assert _find_multiple(1, 256) == 256


def test_rms_norm_shape():
    dim = 32
    norm = RMSNorm(dim)
    x = torch.randn(2, 5, dim)
    out = norm(x)
    assert out.shape == x.shape


def test_rms_norm_scaled_by_weight():
    dim = 8
    norm = RMSNorm(dim)
    torch.nn.init.constant_(norm.weight, 2.0)
    x = torch.ones(1, 1, dim)
    out = norm(x)
    # RMS of all-ones is 1, so output = weight * x / rms = 2 * 1 = 2
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5)


def test_casted_linear_output_shape():
    layer = CastedLinear(16, 32, bias=True)
    x = torch.randn(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 32)


def test_casted_linear_dtype_casting():
    layer = CastedLinear(16, 32, bias=False)
    x = torch.randn(2, 5, 16).to(torch.float16)
    out = layer(x)
    assert out.dtype == torch.float16


def test_casted_layer_norm_shape():
    norm = CastedLayerNorm(32)
    x = torch.randn(2, 5, 32)
    out = norm(x)
    assert out.shape == (2, 5, 32)


def test_casted_layer_norm_dtype_casting():
    norm = CastedLayerNorm(32)
    x = torch.randn(2, 5, 32).to(torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16


def test_swiglu_output_shape():
    hidden = 64
    ffn = SwiGLU(hidden, expansion=4.0)
    x = torch.randn(2, 5, hidden)
    out = ffn(x)
    assert out.shape == (2, 5, hidden)


def test_transformer_output_shape():
    B, S, H = 2, 7, 64
    model = Transformer(hidden_size=H, n_heads=4, num_layers=2, dim_feedforward=128)
    x = torch.randn(B, S, H)
    out = model(cos_sin=None, input=x, pos_enc=None)
    assert out.shape == (B, S, H)


def test_transformer_decoder_output_shape():
    B, S_src, S_tgt, H = 2, 5, 7, 64
    decoder = TransformerDecoder(hidden_size=H, n_heads=4, num_layers=2, dim_feedforward=128)
    source = torch.randn(B, S_src, H)
    target = torch.randn(B, S_tgt, H)
    out = decoder(source=source, target=target, pos_enc=None, key_pos=None)
    assert out.shape == (B, S_tgt, H)


def test_action_reasoning_module_output_shape():
    B, S, H = 2, 6, 64
    arm = ActionReasoningModule(hidden_size=H, n_heads=4, num_layers=2, dim_feedforward=128)
    hidden_states = torch.randn(B, S, H)
    input_injection = torch.randn(B, S, H)
    out = arm(hidden_states=hidden_states, input_injection=input_injection, cos_sin=None, pos_enc=None)
    assert out.shape == (B, S, H)


def test_action_reasoning_module_injection_is_additive():
    B, S, H = 1, 3, 64
    arm = ActionReasoningModule(hidden_size=H, n_heads=4, num_layers=1, dim_feedforward=128)
    arm.eval()
    hidden_states = torch.randn(B, S, H)
    injection = torch.zeros(B, S, H)
    out_zero = arm(hidden_states=hidden_states, input_injection=injection, cos_sin=None, pos_enc=None)

    injection2 = torch.ones(B, S, H)
    out_nonzero = arm(
        hidden_states=hidden_states, input_injection=injection2, cos_sin=None, pos_enc=None
    )
    # With non-zero injection the output should differ
    assert not torch.allclose(out_zero, out_nonzero)
